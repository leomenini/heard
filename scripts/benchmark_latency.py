"""Bare-metal latency probe for heard: per-stage breakdown + Needle probes.

Usage:
  uv run python scripts/benchmark_latency.py
  uv run python scripts/benchmark_latency.py --audio clip.wav   # real STT stage

Stages timed separately (post key-release): audio finalize, STT, intent
(fast classifier vs generative fallback), dispatch. The CONSTRAINED /
UNCONSTRAINED sections are the original Needle-only probes.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from needle import generate

from heard.tools.registry import known_tools, dispatch

MAX_GEN_LEN = 64
SAMPLE_RATE = 16000

PROBES = [
    # system_query
    ("What time is it?", "system_query"),
    ("How's my battery?", "system_query"),
    ("Check the network status", "system_query"),
    ("How much disk space is left?", "system_query"),
    # volume_control
    ("Turn the volume up", "volume_control"),
    ("Turn it down a bit", "volume_control"),
    ("Mute the audio", "volume_control"),
    ("Set the volume to 50 percent", "volume_control"),
    # media_control
    ("Play some music", "media_control"),
    ("Pause the playback", "media_control"),
    ("Skip to the next track", "media_control"),
    ("Go back to the previous song", "media_control"),
    # launch_app
    ("Open Firefox", "launch_app"),
    ("Launch the calculator", "launch_app"),
    ("Start VS Code", "launch_app"),
    # window_action
    ("Close this window", "window_action"),
    ("Make it fullscreen", "window_action"),
    ("Focus on the browser", "window_action"),
    # workspace_switch
    ("Go to workspace 1", "workspace_switch"),
    ("Switch to workspace 3", "workspace_switch"),
]

OFF_TOPIC = [
    "What is the weather tomorrow?",
    "Tell me a joke",
    "Who won the game last night",
    "Remind me to call mom",
]

PROBES_ES = [
    ("qué hora es", "system_query"),
    ("cómo va la batería", "system_query"),
    ("estado de la red", "system_query"),
    ("cuánto espacio hay en el disco", "system_query"),
    ("sube el volumen", "volume_control"),
    ("baja el volumen un poco", "volume_control"),
    ("silencia", "volume_control"),
    ("pon el volumen al cincuenta por ciento", "volume_control"),
    ("reproduce música", "media_control"),
    ("pausa la música", "media_control"),
    ("siguiente canción", "media_control"),
    ("canción anterior", "media_control"),
    ("abre firefox", "launch_app"),
    ("cierra esta ventana", "window_action"),
    ("pantalla completa", "window_action"),
    ("ve al espacio de trabajo tres", "workspace_switch"),
]

OFF_TOPIC_ES = [
    "cuéntame un chiste",
    "qué tiempo hará mañana",
    "recuérdame llamar a mamá",
]


def _pct(latencies):
    latencies = sorted(latencies)
    return {
        "avg": statistics.mean(latencies),
        "p50": statistics.median(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "max": max(latencies),
    }


def _print_stats(label, latencies):
    s = _pct(latencies)
    print(f"  {label:<22} avg {s['avg']:7.1f}ms  p50 {s['p50']:7.1f}ms  "
          f"p95 {s['p95']:7.1f}ms  max {s['max']:7.1f}ms")


def _load_needle():
    from needle import SimpleAttentionNetwork, get_tokenizer, load_checkpoint

    from heard.intent import _checkpoint_file

    ckpt = _checkpoint_file()
    print(f"Loading needle ({ckpt}) ...", end=" ", flush=True)
    t0 = time.perf_counter()
    params, config = load_checkpoint(ckpt)
    model = SimpleAttentionNetwork(config)
    tok = get_tokenizer()
    print(f"{(time.perf_counter() - t0) * 1000:.0f}ms")
    return model, params, tok


def _dry_dispatch():
    """Neutralize subprocess spawns so dispatch timing is pure overhead."""

    def _fake_run(*a, **k):
        class R:
            returncode = 0

        return R()

    def _fake_popen(*a, **k):
        class P:
            pid = 0

        return P()

    subprocess.run = _fake_run
    subprocess.Popen = _fake_popen


def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "  -  "


def bench_intent_stages():
    """Fast-path classify + generative fallback per probe, with accuracy."""
    import numpy as np

    import heard.intent as intent
    from heard.classifier import ACCEPT_SCORE, MARGIN, FLOOR_DECLINE
    from heard.intent import _checkpoint_file

    print("\n--- INTENT STAGES ---")
    print(f"  checkpoint: {_checkpoint_file()}")
    print(f"  thresholds: accept>={ACCEPT_SCORE} margin>={MARGIN} floor<{FLOOR_DECLINE}"
          f"  (override: HEARD_ACCEPT_SCORE / HEARD_MARGIN / HEARD_FLOOR_DECLINE)")
    cls = intent._classifier()
    model, params, tok = _model_triple()
    tools_json = json.dumps(known_tools(), separators=(",", ":"))

    # --- embedding space sanity -------------------------------------------
    keys = list(cls.centroids)
    mat = np.array([[float(np.dot(cls.centroids[a], cls.centroids[b]))
                     for b in keys] for a in keys])
    off_diag = mat[~np.eye(len(keys), dtype=bool)]
    print(f"\n  centroid cross-similarity: mean {off_diag.mean():.3f} "
          f"max {off_diag.max():.3f}")
    if off_diag.max() > 0.98:
        print("  WARNING: centroids nearly identical -- contrastive head looks "
              "degenerate (untrained for retrieval); fast path cannot work.")
    pairs = sorted(((mat[a, b], keys[a], keys[b])
                    for a in range(len(keys)) for b in range(a + 1, len(keys))),
                   reverse=True)[:3]
    print("  closest centroid pairs:")
    for s, a, b in pairs:
        print(f"    {s:.3f}  {a} <-> {b}")

    fast_ms, embed_ms, gen_ms = [], [], []
    fast_ok = fallback_ok = declined_wrong = fallback = misfast = 0
    correct_scores, wrong_scores, top_sims = [], [], []

    def run_one(query: str, expected: str | None) -> str:
        nonlocal fast_ok, fallback_ok, declined_wrong, fallback, misfast
        t0 = time.perf_counter()
        q_emb = cls._encode([query])[0]
        t1 = time.perf_counter()
        verdict = cls.match(query, q_emb=q_emb)
        t2 = time.perf_counter()
        embed_ms.append((t1 - t0) * 1000)

        ranked = cls._rank(query, q_emb=q_emb)   # cheap: reuses embedding
        best_key, best = ranked[0]
        top_sims.append(best)
        same_tool_gap = next(
            (best - s for k, s in ranked[1:] if cls._tool_of[k] != cls._tool_of[best_key]),
            None)
        cross_gap = next(
            (s for k, s in ranked[1:] if cls._tool_of[k] != cls._tool_of[best_key]),
            None)

        tool = verdict.tool_call.get("name") if verdict.tool_call else None

        if verdict.tool_call is not None or verdict.declined:
            fast_ms.append((t2 - t1) * 1000)
            if expected is not None and verdict.declined:
                declined_wrong += 1
                status = "declined(wrong)"
            elif verdict.declined:
                status = "declined"
            elif expected is not None and tool == expected:
                fast_ok += 1
                correct_scores.append(verdict.score)
                status = f"fast ok ({tool})"
            elif expected is None:
                status = "fast WRONG call" if tool else "declined"
            else:
                misfast += 1
                wrong_scores.append(verdict.score)
                status = f"fast WRONG ({tool})"
        else:
            fallback += 1
            t3 = time.perf_counter()
            raw = generate(model, params, tok, query=query, tools=tools_json,
                           stream=False, max_gen_len=MAX_GEN_LEN)
            gen_ms.append((time.perf_counter() - t3) * 1000)
            try:
                calls = json.loads(raw)
                gtool = calls[0].get("name") if isinstance(calls, list) and calls else None
            except ValueError:
                gtool = None
            if expected is None:
                status = "fallback"
            elif gtool == expected:
                fallback_ok += 1
                status = f"fallback ok ({gtool})"
            else:
                status = f"fallback WRONG ({gtool})"

        return (f"  best={best_key:<24} {_fmt(verdict.score)} "
                f"crossgap={_fmt(same_tool_gap)} vs2nd={_fmt(cross_gap)}  "
                f"{status:<24} {query}")

    for query, expected in PROBES:
        print(run_one(query, expected))
    for query, expected in PROBES_ES:
        print(run_one(query, expected))
    for query in OFF_TOPIC:
        print(run_one(query, None))
    for query in OFF_TOPIC_ES:
        print(run_one(query, None))

    n = len(PROBES) + len(PROBES_ES)
    print(f"\n  fast-path accepted {fast_ok}/{n} (en+es), misaccepted {misfast}, "
          f"wrongly declined {declined_wrong}, fell back to needle {fallback}")
    print(f"  fallback accuracy on its share: {fallback_ok}/{fallback}")
    top_sims.sort()
    print(f"  top-centroid sims: min {top_sims[0]:.3f} "
          f"median {statistics.median(top_sims):.3f} max {top_sims[-1]:.3f}")
    if correct_scores:
        print(f"  correct accepts : min {min(correct_scores):.3f} "
              f"median {statistics.median(correct_scores):.3f}")
    if wrong_scores:
        print(f"  MISACCEPTS      : min {min(wrong_scores):.3f} "
              f"(raise ACCEPT_SCORE above this)")
    if fast_ok == 0 and misfast == 0:
        lo = sorted(top_sims)[int(len(top_sims) * 0.10)]
        hi = max(top_sims)
        print(f"  nothing accepted: current ACCEPT_SCORE={ACCEPT_SCORE}; "
              f"probe sims span {lo:.3f}-{hi:.3f}. Try e.g.\n"
              f"    HEARD_ACCEPT_SCORE={max(0.0, hi - 0.02):.2f} "
              f"HEARD_MARGIN={MARGIN / 4:.3f} uv run python scripts/benchmark_latency.py --only intent")
    print("\n  latency:")
    _print_stats("embed (encode query)", embed_ms)
    if fast_ms:
        _print_stats("classify+slots (fast)", fast_ms)
    if gen_ms:
        _print_stats("generative fallback", gen_ms)

    print("\n  generative tail split (where the 6.5s goes):")
    bench_generative_tail(model, params, tok, tools_json)


def bench_generative_tail(model, params, tok, tools_json: str):
    """Split a generative call into encoder prefill vs decoder loop."""
    import jax.numpy as jnp
    from needle.model.architecture import make_padding_mask
    from needle.model.run import _build_encoder_input

    for query in ("Play some music", "Focus on the browser"):
        t0 = time.perf_counter()
        generate(model, params, tok, query=query, tools=tools_json,
                 stream=False, max_gen_len=MAX_GEN_LEN)
        total_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        enc_input = jnp.array([_build_encoder_input(tok, query, tools_json)])
        src_mask = make_padding_mask(enc_input, tok.pad_token_id)
        model.apply({"params": params}, enc_input,
                    src_mask=src_mask, method="encode")
        enc_ms = (time.perf_counter() - t1) * 1000

        enc_tokens = len(_build_encoder_input(tok, query, tools_json))
        print(f"    {query!r:<24} total {total_ms:6.0f}ms  "
              f"encode({enc_tokens} tok) {enc_ms:6.0f}ms  "
              f"decode-loop ~{total_ms - enc_ms:6.0f}ms")


def _model_triple():
    import heard.intent as intent

    intent._classifier()          # ensures model load + encoder JIT
    return intent._model()


def bench_audio_finalize():
    """Concat ring-buffer chunks + VAD trim, the post-release audio work."""
    from heard.stt import _speech_bounds

    print("\n--- AUDIO FINALIZE ---")
    rng = np.random.default_rng(7)
    chunks = []
    pos = 0
    for i in range(40):                       # ~4s at 200ms callback blocks
        n = SAMPLE_RATE // 5
        amp = 0.25 if (i // 4) % 2 == 0 else 0.01   # speech-ish on/off
        chunks.append((pos, (rng.standard_normal(n) * amp).astype(np.float32)))
        pos += n

    def finalize():
        audio = np.concatenate([c for _, c in chunks])
        bounds = _speech_bounds(audio)
        return audio[bounds[0]:bounds[1]] if bounds else audio

    finalize()                                 # warm
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        finalize()
        latencies.append((time.perf_counter() - t0) * 1000)
    _print_stats("finalize 4s buffer", latencies)


def bench_stt(audio_path: str | None):
    """Full-clip vs pause-trimmed-tail transcription."""
    import heard.stt as stt

    print("\n--- STT ---")
    stt.warmup()
    model = stt._model()

    if audio_path:
        import wave
        with wave.open(audio_path, "rb") as w:
            assert w.getframerate() == SAMPLE_RATE, f"need {SAMPLE_RATE}Hz wav"
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            audio = (audio / 32768.0).astype(np.float32)
        label = Path(audio_path).name
    else:
        rng = np.random.default_rng(3)
        speech = (rng.standard_normal(SAMPLE_RATE * 2) * 0.2 * (
            np.hanning(SAMPLE_RATE * 2) ** 0.25)).astype(np.float32)
        gap = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        tail = (rng.standard_normal(SAMPLE_RATE // 2) * 0.005).astype(np.float32)
        audio = np.concatenate([speech, gap, tail])
        label = "synthetic 3s"

    cut = stt.trailing_silence_start(audio)
    full_part = audio[:cut] if cut is not None else audio
    tail_part = audio[cut:] if cut is not None else audio[len(audio) // 2:]

    kw = dict(language="en", beam_size=1, temperature=0.0,
              condition_on_previous_text=False, without_timestamps=True)

    for name, seg in (("full clip", full_part), ("tail only", tail_part)):
        model.transcribe(seg, **kw)            # warm this length bucket
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            segments, _ = model.transcribe(seg, **kw)
            text = " ".join(s.text for s in segments).strip()
            latencies.append((time.perf_counter() - t0) * 1000)
        _print_stats(f"{name} ({len(seg)/SAMPLE_RATE:.1f}s)", latencies)
        print(f"      -> {text!r}"[:90])
    print(f"  input: {label}; streaming hold finalizes pre-pause audio "
          f"during the press, so release pays 'tail only'.")


def bench_dispatch():
    """Registry validate + handler entry cost with spawns neutralized."""
    print("\n--- DISPATCH (dry) ---")
    real_run, real_popen = subprocess.run, subprocess.Popen
    _dry_dispatch()
    calls = [
        {"name": "media_control", "arguments": {"action": "play"}},
        {"name": "volume_control", "arguments": {"action": "set", "amount": "50"}},
        {"name": "workspace_switch", "arguments": {"workspace": "3"}},
        {"name": "launch_app", "arguments": {"app": "firefox"}},
    ]
    for call in calls:
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            dispatch(call)
            latencies.append((time.perf_counter() - t0) * 1000)
        args = ", ".join(f"{k}={v!r}" for k, v in call["arguments"].items())
        _print_stats(f"{call['name']}({args})", latencies)

    print("\n  real spawn check (read-only commands):")
    subprocess.run, subprocess.Popen = real_run, real_popen
    for cmd in (["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
                ["hyprctl", "activewindow"]):
        try:
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, timeout=2)
            ms = (time.perf_counter() - t0) * 1000
            print(f"    {' '.join(cmd)}: {ms:.1f}ms rc={r.returncode}")
        except Exception as e:
            print(f"    {' '.join(cmd)}: unavailable ({type(e).__name__})")


def run_constrained(model, params, tok, tools_json, label: str, constrained: bool):
    print(f"\n--- {label} ---")
    latencies = []
    correct_tool = 0
    parseable = 0

    for q, expected in PROBES:
        t0 = time.perf_counter()
        raw = generate(model, params, tok, query=q, tools=tools_json,
                       stream=False, max_gen_len=MAX_GEN_LEN,
                       constrained=constrained)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        try:
            calls = json.loads(raw)
            parseable += 1
            if calls and isinstance(calls, list):
                correct_tool += int(calls[0].get("name") == expected)
        except (ValueError, TypeError, IndexError):
            pass

        print(f"  {ms:8.0f}ms  {q}")

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    avg = statistics.mean(latencies)
    print(f"\n{label} — {len(latencies)} probes")
    print(f"  avg {avg:.0f}ms  p50 {p50:.0f}ms  p95 {p95:.0f}ms  max {max(latencies):.0f}ms")
    print(f"  parseable {parseable}/{len(PROBES)}  correct-tool {correct_tool}/{len(PROBES)}")
    return latencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None, help="16kHz mono wav for the STT stage")
    ap.add_argument("--skip-legacy", action="store_true",
                    help="skip constrained/unconstrained generative sweeps")
    ap.add_argument("--only", choices=["intent", "stt", "dispatch", "needle"],
                    default=None)
    args = ap.parse_args()

    tools_json = json.dumps(known_tools(), separators=(",", ":"))

    if args.only in (None, "intent"):
        print("=" * 64)
        print("STAGE BENCHMARK")
        print("=" * 64)
        bench_intent_stages()
    if args.only in (None, "stt"):
        bench_audio_finalize()
        bench_stt(args.audio)
    if args.only in (None, "dispatch"):
        bench_dispatch()
    if args.only in (None, "needle") and not args.skip_legacy:
        model, params, tok = _load_needle()
        print("Warmup ...", end=" ", flush=True)
        t0 = time.perf_counter()
        generate(model, params, tok, query="test", tools=tools_json,
                 stream=False, max_gen_len=MAX_GEN_LEN)
        print(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        c = run_constrained(model, params, tok, tools_json,
                            "CONSTRAINED", constrained=True)
        u = run_constrained(model, params, tok, tools_json,
                            "UNCONSTRAINED", constrained=False)

        print("\n--- COMPARISON ---")
        for label, lat in [("constrained  ", c), ("unconstrained", u)]:
            s = _pct(sorted(lat))
            print(f"  {label}:  avg {s['avg']:.0f}ms  p50 {s['p50']:.0f}ms  "
                  f"p95 {s['p95']:.0f}ms  max {s['max']:.0f}ms")


if __name__ == "__main__":
    main()
