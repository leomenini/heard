# TLDR

**heard** is a voice-controlled tooling daemon for Linux. You hold a key, say
"turn the volume up", and it runs the tool. Local-first: the always-on path
never touches the network.

```
hold key -> speech -> intent -> tool call -> done
```

v1.4.0 · Python 3.13 · `uv` · ~3.2k lines · 26 test files · MIT · targets Arch

---

## The one idea

Routing spoken commands to tool calls is **classification, not reasoning**. So
the normal path never runs a language model at all:

| Path | Cost | How often |
|---|---|---|
| Embed + nearest-centroid + regex slots | ~50ms | the vast majority |
| Constrained Needle decode (26M params) | ~6.5s | 1-3 of 20 commands |
| Decline as off-topic | ~50ms | anything unrelated |

Generation is the **fallback**, not the path. Everything else in the repo
follows from that.

---

## Pipeline

```
evdev PTT hold          stt.py     one PortAudio stream stays open for the
  |                                process lifetime; a keypress just marks
  |                                a buffer offset
  v
whisper (base, int8)    stt.py     segments ending in a >=180ms pause are
  |                                transcribed DURING the hold, in a worker
  |                                thread; release pays only the tail
  v
intent.resolve()        intent.py  fast path first, generative second
  |
  +-- classifier.py               bilingual EN/ES prototype centroids,
  |                               cosine rank, threshold gates, slot regex
  +-- needle generate             constrained decode, fallback only
  v
registry.dispatch()     tools/     validate -> handler -> Ok|Rejected|Failed
```

Two trigger sources share one `HoldController` behind a lock: the evdev PTT
key, and `heard hold down/up` over a unix socket (for compositor keybindings).
They can never record simultaneously.

---

## Modules

| File | Lines | What it does |
|---|---|---|
| `classifier.py` | 522 | The heart. ~19 prototype classes, centroids, accept/decline gates, number and app-phrase extraction |
| `stt.py` | 431 | `AudioBus` persistent capture, streaming hold STT, RMS-based VAD, evdev PTT device matching by keycode capability |
| `cli.py` | 220 | typer: `listen`, `hold down/up`, `config get/set/show`. All heavy imports lazy |
| `llm_query.py` | 213 | `question`/`pregunta` branch: OpenAI-compatible endpoint + local-first TTS chain |
| `embedder.py` | 182 | needle-contrastive vs ONNX MiniLM, auto-detects a dead contrastive head |
| `tools/registry.py` | 152 | Single source of truth: model-facing schemas are *generated* from the same table that validates dispatch |
| `intent.py` | 138 | Fast-then-generative orchestration, checkpoint resolution |
| `ipc.py` | 117 | Unix socket at `$XDG_RUNTIME_DIR/heard.sock`, chmod 600, `ping`/`down`/`up` |
| `config.py` | 100 | Flat typed TOML store at `~/.config/heard/config.toml` |
| `lock.py` | 50 | Single-instance flock'd pidfile; dies with the process, so stale files are harmless |
| `events.py` | 32 | Local JSONL log at `~/.local/state/heard/events.jsonl`. Never leaves the machine |
| `tools/helpers/wm.py` | 361 | Hyprland / Sway / KDE / GNOME abstraction |

---

## Tools

| Tool | Backend | Notes |
|---|---|---|
| `launch_app` | WM + `uwsm`/spawn | Focus-or-launch; spoken names fuzzy-matched against installed `.desktop` entries |
| `media_control` | MPRIS via jeepney | One persistent D-Bus connection, `playerctl` fallback |
| `volume_control` | `wpctl` | Percentages parsed from speech: "fifty", "65%", "half" |
| `microphone_control` | `wpctl` | Mute/unmute default source |
| `screen_record` | `wf-recorder` / `ffmpeg x11grab` | Pidfile state so `stop` finalizes the file |
| `window_action` | WM | Close, focus, fullscreen. Focus targets need generation |
| `workspace_switch` | WM | "workspace three" -> 3 |
| `system_query` | `/sys` + stdlib | Battery, time, network, disk |

---

## Design decisions worth knowing

- **`unknown:x` is a real centroid.** "Tell me a joke" is declined in
  milliseconds by winning an explicit off-topic class, not sent anywhere.
- **The fast path needs no Needle weights.** The bundled checkpoint's
  contrastive head is dead, so embedding runs on fastembed MiniLM; the
  checkpoint is loaded lazily and only the rare generative fallback wants it.
- **Schemas are generated, not stored.** No static schema file means no drift
  between what the model sees and what dispatch accepts.
- **Work moved out of the tail is felt latency.** The streaming-during-hold
  design exists entirely to make key release cheap.
- **Thresholds come from benchmarks, not feel.** Misaccepts are the hard gate;
  all four gates are env-overridable for tuning.
- **`Rejected` vs `Failed`** is a real distinction: input was wrong, versus the
  world was wrong (binary missing, D-Bus error). Handlers never raise.
- **Lazy imports everywhere** keep whisper/needle/jax off the startup path.
- **Tests can't touch your session.** An autouse fixture makes any unmocked WM
  call fail loudly instead of closing your windows.

---

## Network surface (the complete list)

| Feature | Network? |
|---|---|
| STT / intent / dispatch | **never** |
| Model downloads | first run only (~200MB into local caches) |
| TTS | piper is local; the `gtts` extra is online and off by default |
| `question ...` query mode | yes, opt-in, requires you to configure `llm_url` |

---

## Sharp edges

- The bundled Needle checkpoint has an **untrained contrastive head**, so the
  fast path actually runs on fastembed MiniLM. Needle only does the rare
  generative decode.
- **The benchmark probe set overstates accuracy.** Its `launch_app` probes are
  near-verbatim prototype utterances, so they pass by construction. On held-out
  app names, `launch_app` accepts 3/12 — spoken names pull toward
  `window_action:close`. An opt-in `HEARD_DICT_RESCUE` gate takes that to 9/12
  with no misaccepts; see CLAUDE.md for the tuning table.
- `checkpoints/` is gitignored and ships no weights. Running weightless works,
  but everything routed to the generative fallback declines — including
  `window_action:focus`, which always routes there by design.
- Window tools cover Hyprland, Sway, KDE, GNOME, plus a generic `x11`
  (wmctrl/EWMH) backend for Cinnamon, XFCE, MATE, i3. Wayland compositors
  outside that list still raise.
- The generative fallback re-forwards its full prefix every token (no KV
  cache). That's the ~6.5s. See `reports/fallback_speed.md`.
- EN/ES only. More languages need prototypes plus a multilingual embedder
  entry; the mechanism exists.
- Key capture needs the `input` group, and fresh membership needs a new
  session.

---

## Getting around

```bash
uv sync --dev && uv run heard listen     # run it
uv run pytest                            # test it
uv run ruff check . && uv run mypy heard # the other two CI gates

scripts/benchmark_latency.py             # per-stage timings -> reports/latency.md
scripts/stt_probe.py                     # find fastest whisper size/threads here
scripts/train_needle_head.py             # finetune the contrastive head
scripts/export_fallback.py               # the KV-cache/ONNX spike
```

`README.md` is the full manual · `CLAUDE.md` is the working guide for agents ·
`CHANGELOG.md` is the history · `reports/` holds measured numbers.
