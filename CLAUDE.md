# CLAUDE.md

Voice-controlled tooling daemon for Linux: push-to-talk speech to intent
resolution to tool dispatch. Local-first, Python 3.13, `uv`-managed.

## Commands

```bash
uv sync --dev                       # deps (first run also pulls whisper/MiniLM weights)
uv run pytest                       # full suite
uv run pytest tests/test_wm.py -v   # one file
uv run ruff check .                 # lint (line-length 100)
uv run mypy heard                   # type check
uv run heard listen                 # run the daemon in the foreground
```

CI runs exactly `ruff check .`, `mypy heard`, `pytest` on push/PR to master.
All three must pass before a change is done.

Benchmarks live in `scripts/`; results belong in `reports/`, not in commit
messages:

```bash
uv run python scripts/benchmark_latency.py --only intent   # fast vs generative split
uv run python scripts/benchmark_latency.py --only stt
```

## Architecture

```
evdev PTT hold -> AudioBus -> whisper (streaming, pause-delimited)
  -> intent.resolve()
       ├─ classifier.py  nearest-centroid over embeddings  (~50ms, the normal path)
       └─ needle generate (constrained)                    (~6.5s, fallback only)
            -> tools/registry.py -> handler -> Ok|Rejected|Failed
```

The central design decision: **generation is the fallback, not the path.**
`classifier.py` holds bilingual EN/ES prototype utterances per class (one per
tool, or per enumerable action value), embedded once at warmup into centroids.
A transcript is embedded, cosine-ranked, and accepted only if it clears
`ACCEPT_SCORE` with a `MARGIN` lead over the best other-tool class. Slots that
cannot be classified (numbers, app names) come from regex/lookup
(`extract_percent`, `parse_spoken_number`, `extract_app_phrase`), never from
generation.

Load-bearing details that are easy to break:

- **`unknown:x` is a real centroid.** Off-topic input ("tell me a joke") is
  declined in milliseconds by winning that class. Do not treat it as an
  ordinary tool class in ranking logic.
- **`FALLBACK_CLASSES`** deliberately never fast-accepts; those classes need
  generated slots. Adding a class here is how you say "this needs the model".
- **`launch_app` gets a lower bar** (`DICT_FLOOR`) when a strict rapidfuzz hit
  against installed `.desktop` entries corroborates it, guarded by a token
  overlap check because `token_set_ratio` scores 100 on subsets.
- **Thresholds are tuned from benchmark output, not feel.** Misaccepts are the
  hard gate: a change that raises accept rate while introducing a misaccept is
  a regression. All are env-overridable (`HEARD_ACCEPT_SCORE`, `HEARD_MARGIN`,
  `HEARD_FLOOR_DECLINE`, `HEARD_DICT_FLOOR`, `HEARD_RESCUE_FLOOR`).
- **`_dict_rescue()` is opt-in (`HEARD_DICT_RESCUE`), and stays that way.**
  Spoken app names ("open brave") pull toward `window_action:close`, so
  `launch_app` loses the ranking and the corroboration gate — which requires it
  to *win* — never fires. The rescue lets it place within `RESCUE_RANK` instead.
  It has its own `RESCUE_FLOOR` deliberately: sharing `DICT_FLOOR` would mean
  tuning the rescue silently loosens the corroboration gate above it. The
  measured numbers below are fastembed geometry and do not transfer to a
  trained Needle head (`accept_score` 0.85 vs 0.55).

`tools/registry.py` is the single source of truth: the schemas shown to the
model are *generated* from the same `REGISTRY` that validates dispatch. Never
hand-write a tool schema elsewhere; there is no static schema file, and that is
on purpose.

`stt.py` keeps one PortAudio stream open for the process lifetime, so a PTT
press only marks a buffer offset. `HoldSession` transcribes pause-delimited
segments in a worker thread *during* the hold, so key release pays only the
tail. Keep that property: work moved from the hold into the tail is directly
felt latency.

`ipc.py` and the evdev loop share one `HoldController` behind a lock, so the
two trigger sources can never record simultaneously. Use the public
`is_busy()` / `cancel_pending()`; do not poke private session attributes.

## Conventions

- **Lazy imports inside functions are deliberate**, not sloppiness: they keep
  heavy deps (whisper, needle, jax, fastembed) off the startup path. `PLC0415`
  is ignored repo-wide for `heard/**` and `scripts/**` for this reason.
- **Every `ruff` ignore in `pyproject.toml` carries a justification comment.**
  If you add one, explain why the pattern is intended. If you find yourself
  wanting a broad new ignore, that's a signal to reconsider the code.
- **Handlers return `Ok | Rejected | Failed`, never raise to the caller.**
  `Rejected` means "input was wrong" (bad args, declined); `Failed` means "the
  world was wrong" (binary missing, D-Bus error). Broad `except Exception`
  around hardware/D-Bus/IO edges is intentional (`BLE001` ignored).
- Config is a flat typed store in `config.py`; a new key needs an entry in
  `DEFAULTS`, and a validated key needs its `VALID_*` tuple checked in
  `set_key`.
- Prose in README/CHANGELOG uses no em dashes. Keep that.

## Testing

`tests/conftest.py` mocks `evdev`, `sounddevice`, `faster_whisper`, and
`needle` at module level so imports never touch hardware or load models. An
autouse fixture blocks `wm._run`, `wm._kde_kdotool`, and `wm._dbus_call` — a
test that reaches the live compositor fails loudly rather than closing the
developer's windows. **Never disable that fixture**; mock those three
explicitly in tests that need WM behavior.

Tests are class-grouped (`class TestVolumeControl:`) with `mock.patch` on the
subprocess boundary of the module under test.

### Benchmarking the classifier — read before trusting a number

**The probe set in `scripts/benchmark_latency.py` cannot detect `launch_app`
regressions.** Its three English `launch_app` probes ("Open Firefox", "Launch
the calculator", "Start VS Code") are near-verbatim utterances from the
prototype bank in `classifier.py`, so they pass by construction. The headline
"19/20 accepted, 0 misaccepts" is therefore measured partly against the
classifier's own training phrases.

Measured on held-out phrasings ("open <real installed app>", names absent from
the prototypes) with fastembed MiniLM — 12 launch probes, 19 off-topic:

| config | launch accepted | misaccepts | off-topic misaccepts |
|---|---|---|---|
| baseline (rescue off) | 3/12 | 0 | 0 |
| rank<=3, floor 0.35 | 7/12 | 0 | 0 |
| **rank<=3, floor 0.30** | **9/12** | **0** | **0** |
| rank<=3, floor 0.25 | 11/12 | 0 | 2 |
| rank<=5, floor 0.30 | 9/12 | 0 | 1 |

`RESCUE_RANK` is nearly inert; `RESCUE_FLOOR` is the load-bearing knob, and the
cliff is sharp. Both failures at 0.25 are the same shape — a sentence
*mentioning* an app read as a command to launch it (`"add vlc to my shopping
list"` -> `launch_app`).

So when touching the classifier: build a probe set of phrasings **not** in
`PROTOTYPES`, and always count off-topic misaccepts alongside accept rate.
Note `bench_intent_stages()` calls `_checkpoint_file()` and loads Needle for
its generative leg, so it cannot run without weights — a fast-path-only harness
is needed on machines with no checkpoint.

## Known sharp edges

- The bundled Needle checkpoint has an **untrained contrastive head** (decayed
  to zero by pretraining), so the fast path actually runs on int8 ONNX MiniLM
  via fastembed. `HEARD_EMBEDDER=needle|fastembed` forces a backend.
- `checkpoints/` is gitignored and the repo ships no weights. The daemon runs
  without them: `resolve_encoder` takes a `needle_factory` **callable**, so
  `auto` degrades to fastembed instead of dying at warmup, and
  `_generate_resolve` returns a declining `Resolution` rather than raising when
  the checkpoint is missing. **Keep both properties** — an exception on either
  path kills `heard listen`, since `cli.py` does not catch inside `handle()`.
  Cost of running weightless: anything routed to the fallback declines, which
  includes every `FALLBACK_CLASSES` member (`window_action:focus`).
- Window/workspace tools cover Hyprland, Sway, KDE, GNOME, and a generic
  `x11` backend (wmctrl/EWMH) that catches Cinnamon, XFCE, MATE, i3.
  `wm.detect()` tries `x11` **last** so richer backends win on X11 sessions.
  Every public function in `wm.py` dispatches on `require()` with an explicit
  branch per backend and raises `_unsupported()` at the end — never add a bare
  `else`, or a new backend silently inherits another one's commands.
- `launch_app` tries `uwsm app -- <binary>` first, then a plain spawn, so it
  works on both wlroots sessions and X11 desktops.
- The generative fallback re-forwards its full prefix every token (no KV
  cache). See `reports/fallback_speed.md`; KV-caching is the identified fix.
- Key capture needs membership in the `input` group; fresh group membership
  requires a new session (`sg input -c '...'` to test immediately).
