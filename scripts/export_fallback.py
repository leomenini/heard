"""Fallback-speed spike: where the ~6.5s generative tail goes, and what fixes it.

Usage (needs a Needle checkpoint; see intent._checkpoint_file):
  uv run python scripts/export_fallback.py                 # baseline + step profile
  uv run python scripts/export_fallback.py --onnx          # + export attempt
  uv run python scripts/export_fallback.py --onnx-install  # prints extra deps

Findings feed reports/fallback_speed.md. Nothing here changes production
paths -- this is measurement + feasibility only.

Three measurements:
  baseline     end-to-end generate() latency on probe queries
  step profile wall time of every decoder forward across the generation,
               showing how per-step cost grows with sequence length
               (needle re-forwards the whole prefix each token: no KV cache)
  onnx         export the decoder-step function to ONNX (jax2onnx if
               available, else jax2tf+tf2onnx), quantize int8 dynamically,
               and compare per-step latency against JAX
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_GEN_LEN = 64
TOOLS_JSON_MIN = '{"name":"media_control","parameters":{"action":{"enum":' \
                 '["play","pause"]}}}'
PROBES = ["Play some music", "Focus on the browser", "Set volume to fifty percent"]


def _load():
    from heard import intent

    intent._classifier()                      # loads weights + warms encoders
    return intent._model()


def _pct(xs):
    xs = sorted(xs)
    return statistics.median(xs), xs[int(len(xs) * 0.95)]


def baseline(model, params, tok) -> None:
    from needle import generate

    print("\n## Baseline generate()")
    for q in PROBES:
        generate(model, params, tok, query=q, tools=TOOLS_JSON_MIN,
                 stream=False, max_gen_len=MAX_GEN_LEN)      # JIT warm
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            generate(model, params, tok, query=q, tools=TOOLS_JSON_MIN,
                     stream=False, max_gen_len=MAX_GEN_LEN)
            runs.append((time.perf_counter() - t0) * 1000)
        p50, p95 = _pct(runs)
        print(f"  {q!r:<36} p50 {p50:6.0f}ms  p95 {p95:6.0f}ms")


def step_profile(model, params, tok) -> list[float]:
    """Per-decode-forward times for one generation (quadratic evidence)."""
    import jax.numpy as jnp
    from needle.model.architecture import make_padding_mask
    from needle.model.run import _build_encoder_input, _get_decode_fn

    print("\n## Decoder step profile (per-forward wall time)")
    rows: list[float] = []
    for q in PROBES[:1]:
        enc_tokens = _build_encoder_input(tok, q, TOOLS_JSON_MIN)
        enc_input = jnp.array([enc_tokens])
        src_mask = make_padding_mask(enc_input, tok.pad_token_id)
        t0 = time.perf_counter()
        encoder_out, enc_mask = model.apply(
            {"params": params}, enc_input, src_mask=src_mask, method="encode")
        enc_ms = (time.perf_counter() - t0) * 1000

        decode_fn = _get_decode_fn(model, MAX_GEN_LEN)
        buf = jnp.full((1, MAX_GEN_LEN), tok.pad_token_id, dtype=jnp.int32)
        buf = buf.at[0, 0].set(tok.eos_token_id)

        decode_fn(params, buf, encoder_out, enc_mask)           # JIT warm
        steps = []
        for i in range(MAX_GEN_LEN - 1):
            t1 = time.perf_counter()
            logits = decode_fn(params, buf, encoder_out, enc_mask)
            steps.append((time.perf_counter() - t1) * 1000)
            next_tok = int(jnp.argmax(logits[0, i]))
            if next_tok == tok.eos_token_id:
                break
            buf = buf.at[0, i + 1].set(next_tok)

        rows = steps
        n = len(steps)
        first = statistics.median(steps[: max(3, n // 8)])
        last = statistics.median(steps[-max(3, n // 8):])
        total_enc = len(enc_tokens)
        print(f"  {q!r}: enc {enc_ms:.0f}ms ({total_enc} tok) -> "
              f"{n} decode forwards")
        print(f"    first-third p50 {first:.1f}ms/step   "
              f"last-third p50 {last:.1f}ms/step   "
              f"growth x{last / max(first, 1e-9):.2f}")
        print(f"    decode total {sum(steps):.0f}ms "
              f"(KV cache would cut this to roughly n x {first:.1f}ms "
              f"~= {n * first:.0f}ms)")
    return rows


def _export_decoder_step(model, params, tok, out_dir: Path) -> Path | None:
    """Export the decoder-step function; returns an .onnx path or None."""
    import jax
    import jax.numpy as jnp
    from needle.model.architecture import make_padding_mask
    from needle.model.run import _build_encoder_input, _get_decode_fn

    enc_tokens = _build_encoder_input(tok, PROBES[0], TOOLS_JSON_MIN)
    enc_input = jnp.array([enc_tokens])
    src_mask = make_padding_mask(enc_input, tok.pad_token_id)
    encoder_out, enc_mask = model.apply({"params": params}, enc_input,
                                        src_mask=src_mask, method="encode")
    decode_fn = _get_decode_fn(model, MAX_GEN_LEN)

    def step(buf, enc_out, mask):
        return decode_fn(params, buf, enc_out, mask)

    jitted = jax.jit(step)
    buf = jnp.full((1, MAX_GEN_LEN), tok.pad_token_id, dtype=jnp.int32)
    args = (buf, encoder_out, enc_mask)
    jitted(*args)                                             # compile + shape check

    enc_len, d_model = encoder_out.shape[1], encoder_out.shape[2]
    print(f"\n## ONNX export (decoder step, fixed shapes "
          f"buf=(1,{MAX_GEN_LEN}) enc=({enc_len},{d_model}))")

    try:                                                      # strategy A: jax2onnx
        import jax2onnx

        target = out_dir / "decoder_step.onnx"
        jax2onnx.convert(jitted, list(args), str(target))
        print(f"  jax2onnx export ok -> {target}")
        return target
    except ImportError:
        print("  jax2onnx not installed; trying jax2tf + tf2onnx ...")
    except Exception as e:
        print(f"  jax2onnx failed: {type(e).__name__}: {e}")

    try:                                                      # strategy B: tf2onnx
        import tensorflow as tf
        import tf2onnx
        from jax.experimental import jax2tf

        tf_fn = jax2tf.convert(step, with_gradient=False)
        specs = [
            tf.TensorSpec((1, MAX_GEN_LEN), tf.int32),
            tf.TensorSpec(tuple(encoder_out.shape), tf.float32),
            tf.TensorSpec(tuple(enc_mask.shape), tf.float32),
        ]
        target = out_dir / "decoder_step.onnx"
        tf2onnx.convert.from_function(tf_fn, input_signature=specs, opset=17,
                                      output_path=str(target))
        print(f"  tf2onnx export ok -> {target}")
        return target
    except ImportError as e:
        print(f"  tf2onnx route unavailable ({e.name}); install with:")
        print("    uv pip install tensorflow-cpu tf2onnx onnx onnxruntime")
    except Exception as e:
        print(f"  tf2onnx export failed: {type(e).__name__}: {e}")
    return None


def onnx_compare(model, params, tok, onnx_path: Path) -> None:
    """int8-dynamic-quantize the exported step and time it against JAX."""
    import jax.numpy as jnp
    import onnxruntime as ort
    from needle.model.architecture import make_padding_mask
    from needle.model.run import _build_encoder_input, _get_decode_fn

    quant = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(onnx_path), str(quant), weight_type=QuantType.QInt8)
        print(f"  int8 dynamic quantization -> {quant}")
    except Exception as e:
        print(f"  quantization failed ({type(e).__name__}: {e}); using fp32 ONNX")
        quant = onnx_path

    enc_tokens = _build_encoder_input(tok, PROBES[0], TOOLS_JSON_MIN)
    enc_input = jnp.array([enc_tokens])
    src_mask = make_padding_mask(enc_input, tok.pad_token_id)
    encoder_out, enc_mask = model.apply({"params": params}, enc_input,
                                        src_mask=src_mask, method="encode")
    np_buf = np.full((1, MAX_GEN_LEN), tok.pad_token_id, dtype=np.int32)
    candidates = [np_buf, np.asarray(encoder_out), np.asarray(enc_mask)]
    sess = ort.InferenceSession(str(quant), providers=["CPUExecutionProvider"])
    feeds = {inp.name: arr
             for inp, arr in zip(sess.get_inputs(), candidates, strict=True)}
    sess.run(None, feeds)                                     # warm

    ort_ms = []
    for _ in range(20):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        ort_ms.append((time.perf_counter() - t0) * 1000)

    decode_fn = _get_decode_fn(model, MAX_GEN_LEN)
    jax_ms = []
    for _ in range(20):
        t0 = time.perf_counter()
        decode_fn(params, jnp.array(np_buf), encoder_out, enc_mask)
        jax_ms.append((time.perf_counter() - t0) * 1000)

    o50, o95 = _pct(ort_ms)
    j50, j95 = _pct(jax_ms)
    print("\n### Per-decode-step: ONNX vs JAX (early steps, same shapes)")
    print("| backend | p50 | p95 |")
    print("|---|---|---|")
    print(f"| jax (fp32, jit) | {j50:.1f}ms | {j95:.1f}ms |")
    print(f"| onnx ({quant.stem}) | {o50:.1f}ms | {o95:.1f}ms |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", action="store_true",
                    help="also attempt ONNX export + int8 quantization")
    args = ap.parse_args()

    try:
        from heard.intent import _checkpoint_file

        ckpt = _checkpoint_file()
    except RuntimeError as e:
        print(f"needle checkpoint required:\n{e}")
        sys.exit(1)
    print(f"checkpoint: {ckpt}")

    model, params, tok = _load()

    baseline(model, params, tok)
    step_profile(model, params, tok)

    if args.onnx:
        out_dir = Path("reports")
        out_dir.mkdir(exist_ok=True)
        onnx_path = _export_decoder_step(model, params, tok, out_dir)
        if onnx_path is not None:
            onnx_compare(model, params, tok, onnx_path)


if __name__ == "__main__":
    main()
