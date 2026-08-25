# heard

> Voice-controlled tooling daemon for Linux. Speech to intent resolution to tool dispatch. Local-first.

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white&style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-10AC84?style=for-the-badge)]()
[![Arch](https://img.shields.io/badge/Target-Arch%20Linux-1793D1?logo=archlinux&logoColor=white&style=for-the-badge)]()
[![CI](https://img.shields.io/github/actions/workflow/status/anomalyco/heard/test.yml?style=for-the-badge&logo=github)]()

hear + d. It's the joke `sshd` would make if it could. Reads as an English word, is exactly what the daemon does, and `heard: launching spotify` in a log line is funny.

> **v1.4.** `heard listen` is both the foreground loop and the daemon: run it directly, or install the systemd unit and drive holds from compositor keybindings. Ctrl-C stops the foreground loop.

---

## Quick start

```bash
uv sync                          # one-time: deps + tokenizer/model downloads
uv run heard listen
```

If `/dev/input` permission errors appear (fresh group memberships need a new session):

```bash
sg input -c 'uv run heard listen'    # or log out and back in after `usermod -aG input $USER`
```

Hold **Left Shift**, speak, release. Output looks like:

```
heard: ready in 21.4s
heard: language=en  ptt=KEY_LEFTSHIFT via AT Translated Set 2 keyboard
heard: listening (hold to talk, release to send)
  you: volume up
  heard: volume_control(action='up')
  heard: Ok(tool='volume_control', detail='volume up')  [hold 1.8s | stt tail 240ms | intent 58ms | dispatch 41ms fast 0.71]
```

Spanish: `uv run heard config set language es`: switches STT decoding to
Spanish and swaps the embedding model to a cross-lingual one; the classifier
understands both languages either way.

---

## Daemon & compositor keybindings

`heard listen` doubles as the daemon: besides watching your PTT key it serves
a unix socket (`$XDG_RUNTIME_DIR/heard.sock`) so keybindings can drive holds
as one-shot commands. Install the systemd user unit:

```bash
mkdir -p ~/.config/systemd/user
cp packaging/systemd/heard.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now heard
journalctl --user -u heard -f          # watch transcripts + results live
```

Then bind hold-down/hold-up to any key. The binding pair uses press and
release events of the same key:

**Hyprland** (`hyprland.conf`; bind = on press, bindr = on release):

```ini
bind  = , mouse:275, exec, heard hold down    # example: side mouse button
bindr = , mouse:275, exec, heard hold up
```

**Sway** (`config`):

```
bindsym --release XF86AudioMicMute exec heard hold up
bindsym XF86AudioMicMute exec heard hold down
```

**KDE**: System Settings → Shortcuts → Add Command, `heard hold down` on
press and `heard hold up` on release of the chosen shortcut.

**Cinnamon/X11 note**: Cinnamon custom shortcuts fire on press only and cannot
express the press/release pair, so compositor-driven holds are unavailable
there. The evdev PTT path is the primary trigger on these desktops.

The daemon keeps models hot, so a binding-triggered command costs zero load
time; both triggers share one controller and can never record simultaneously.

Every command prints a per-stage breakdown: `hold` is your speaking time,
`stt tail` is transcription cost *after* key release (streaming during the
hold means this is only the post-pause remainder), then intent and dispatch.
The trailing tag marks fast-path accepts with their cosine score; its absence
means the generative fallback ran.

Requirements: a microphone, PipeWire (`wpctl`), and membership in the `input`
group for key capture. Window/workspace tools use one of five backends
(Hyprland, Sway, KDE, GNOME, or generic X11 via wmctrl) detected from the
session; the X11 backend is tried last, so Cinnamon, XFCE, MATE, and i3 are
covered without displacing the richer ones. On GNOME, install the
tiny companion Shell extension,
[`packaging/gnome-shell/`](packaging/gnome-shell/),
since stock shells disallow D-Bus Eval; without it heard falls back to Eval,
which only works in unsafe mode.
First run downloads Whisper + MiniLM weights (~200MB total) into local caches.

---

## Query mode (spoken answers)

Transcripts starting with **question** or **pregunta** skip intent resolution
and go to an OpenAI-compatible chat endpoint you configure yourself (opt-in,
the only networked feature besides optional gTTS):

```bash
uv run heard config set llm_url https://api.example.com/v1/chat/completions
uv run heard config set llm_api_key sk-...
uv run heard config set llm_model gpt-4o-mini   # endpoint-dependent
```

The answer prints and is spoken out loud (background thread, daemon stays
responsive). Speech is local-first: piper runs fully offline after its
one-time voice download; auto chain is piper -> gtts -> flite -> console,
pinned via `tts_backend`, disabled entirely with `tts_backend = none`.

---

## How it works

1. Hold left Shift: evdev captures key state (no X11/Wayland dependency). Devices are selected by actual keycode capability, so PTT lands on your keyboard instead of the first "Power Button".
2. Audio streams continuously into a ring buffer; while the key is down, any segment ending in a ≥180ms pause is transcribed in a background thread (faster-whisper base, int8). Release only pays the post-pause tail.
3. Resolve intent: the transcript is embedded and matched against precomputed intent centroids; dispatch in single-digit ms when confident, cheap decline for off-topic, generative Needle decode only when unsure. Transcripts starting with "question"/"pregunta" branch to query mode instead.
4. Dispatch to the matching tool handler (launch, media, volume, microphone, window, workspace, screen record, system query).
5. Print the resolved tool call immediately, then the result with stage timings.

---

## Measured latency (i5-1335U, no GPU)

| Stage | p50 | notes |
|---|---|---|
| intent: embed query (MiniLM int8) | 53ms | p95 101ms incl first-call warmup |
| intent: classify + slots | <1ms | numpy dots over 18 centroids + regex |
| intent: generative fallback | 6.5s | constrained decode, rare by design |
| dispatch (dry) | ~0.1ms | validate + handler entry |
| real spawn (wpctl / hyprctl) | 38 / 19ms | MPRIS tools skip this via persistent D-Bus |
| end-to-end post-release | pending | fill from live `[...]` lines into [`reports/latency.md`](reports/latency.md) |

Fast-path accuracy on the probe set: **19/20 accepted** (17/20 measured on
hardware before the dictionary gate), **0 misaccepts**, **0 wrong declines**,
all off-topic rejected via an explicit unknown centroid. The one holdout
("focus on the browser") is deliberately ambiguous until focus-or-launch
semantics absorb it.

Full methodology + tuning tables: [`reports/latency.md`](reports/latency.md).

---

## Why Needle?

heard routes spoken commands to tool calls. That is a classification problem, not a general reasoning problem. Needle, a 26M parameter function-call model, stays at the core for three reasons:

**Speed.** A larger LLM adds seconds of latency on consumer hardware. Generation is now the *fallback*, not the path: the common command resolves through embedding classification and slot lookup in well under 100ms; only ambiguous phrasings pay the ~6.5s decode. The bundled checkpoint's contrastive head was decayed to zero by pretraining, so the embeddings come from an int8 ONNX MiniLM instead (`embedder.py` picks this automatically; `HEARD_EMBEDDER=needle|fastembed` forces one).

**Local-first.** No cloud APIs, no network round trip, no special hardware. Everything runs on a laptop CPU. The only always-on path is fully offline; the features below are the complete, explicit list of what can touch the network:

| Feature | Network? | Notes |
|---|---|---|
| STT / intent / dispatch | never | models load from local caches |
| TTS (query answers) | only if piper voice not yet cached | piper runs locally; `gtts` extra is online and off by default; `tts_backend = flite` is fully offline |
| `question ...` query mode | yes (opt-in) | requires configuring `llm_url` yourself |
| model downloads | first run only | Whisper + MiniLM + piper voices (~200MB total) into local caches |

**Specialization.** A 7B+ general-purpose model dedicates most of its capacity to knowledge and dialogue. Needle is trained specifically for function-call extraction; tool descriptions steer it (tightening them alone moved probe accuracy 75% → 92%). It is not a chat model and never answers questions: open-ended input is declined in milliseconds by the classifier's unknown centroid, not sent anywhere.

---

## Tools

| Tool | Handler | What it does |
|---|---|---|
| `launch_app` | `tools/apps.py` | Focus-or-launch: fuzzy-matches running windows via the WM backend, focuses if already open, else launches through uwsm, falling back to a plain spawn where uwsm is absent. Spoken names resolve against installed `.desktop` entries (rapidfuzz) |
| `media_control` | `tools/media.py` | Play, pause, next, previous over MPRIS: one persistent D-Bus connection (jeepney), playerctl fallback |
| `volume_control` | `tools/volume.py` | Up, down, mute, set percent: percentages parsed from speech ("fifty", "65%", "half") |
| `microphone_control` | `tools/microphone.py` | Mute or unmute the default microphone source via wpctl |
| `screen_record` | `tools/screen_record.py` | Start/stop screen recording; wf-recorder on Wayland, ffmpeg x11grab fallback. Spoken screen labels map to outputs via config; files land in `screenrecord_folder` |
| `window_action` | `tools/window.py` | Close, fullscreen, minimize, focus via the WM backend; minimize is x11-only (Muffin ignores the EWMH hidden state, so the iconify request goes out as ICCCM WM_CHANGE_STATE through python-xlib); focus targets fuzzy-match open windows (generative parsing for targets) |
| `workspace_switch` | `tools/workspace.py` | Switch by number, digits or words ("workspace three") |
| `system_query` | `tools/system_query.py` | Battery, time, network, disk status |

Tool schemas are generated from the registry (not stored as a static file), so descriptions, validation enums, and finetune data all derive from one source. No drift between what the model sees and what dispatch accepts.

---

## Architecture

```
evdev (capability-matched PTT devices)
   └─ hold: ring buffer streams through whisper in the background,
      pause-delimited segments finalize as you speak
        └─ release: transcribe only the tail
             └─ intent.resolve()
                  ├─ embedder.py   needle-contrastive | ONNX MiniLM (auto)
                  ├─ classifier.py nearest-centroid + unknown rejection
                  │                + slot lookup (numbers, app phrases)
                  │                + .desktop dictionary corroboration for launch_app
                  └─ needle generate (constrained)   ← fallback only
                       └─ tools/registry.py -> handler -> stdout
```

The dispatcher never passes raw model output to a shell. Arguments are validated against registry enums; the model selects an allowlisted action (`shell_allowlist.py`), never arbitrary shell text.

### Configuration

`~/.config/heard/config.toml`, managed via `uv run heard config get|set|show`
(`HEARD_CONFIG` overrides the path):

| Key | Default | Purpose |
|---|---|---|
| `language` | `"en"` | `"en"` \| `"es"`: STT decode language + embedder pick; classifier is always bilingual |
| `ptt_key` | `"KEY_LEFTSHIFT"` | any evdev `KEY_*` name |
| `checkpoint` | `""` | Needle weights path override; empty resolves `$HEARD_CHECKPOINT`, then any `checkpoints/*.pkl` |
| `embedder_model` | auto per language | explicit fastembed model override |
| `stt_model_size` | `"base"` | faster-whisper size (`tiny`/`base`/`small`) |
| `stt_cpu_threads` | auto | pin ctranslate2 threads; try 2 on hybrid CPUs |
| `wm_backend` | `"auto"` | `auto` \| `hyprland` \| `sway` \| `kde` \| `gnome` \| `x11` (KDE windows via kdotool, workspaces/fullscreen via KWin scripting; GNOME via the companion Shell extension; x11 via wmctrl) |
| `events` | `true` | local JSONL usage log |
| `tts_enabled` | `true` | speak query answers out loud |
| `tts_backend` | auto | `""`(auto: piper→gtts→flite) \| `piper` \| `gtts` \| `flite` \| `none`: spoken answers for query mode |
| `llm_url` / `llm_api_key` / `llm_model` | empty | OpenAI-compatible endpoint powering query mode |
| `screenrecord_output` / `screenrecord_outputs` / `screenrecord_folder` | `~/Videos` | default output, spoken-label map (`One=DP-1`), save folder for screen_record |

Env vars override config for tuning and debugging:

| Variable | Default | Purpose |
|---|---|---|
| `HEARD_CHECKPOINT` | `checkpoints/needle_checkpoint.pkl` | Needle weights (any `*.pkl` in `checkpoints/` also works) |
| `HEARD_EMBEDDER` | `auto` | `needle` \| `fastembed` \| `auto` |
| `HEARD_EMBED_MODEL` | per language | exact fastembed model name |
| `HEARD_ACCEPT_SCORE` | `0.55` | min cosine to accept a centroid outright |
| `HEARD_MARGIN` | `0.04` | required lead over best other-tool class |
| `HEARD_FLOOR_DECLINE` | `0.35` | below this (+ unknown agreeing): reject without fallback |
| `HEARD_DICT_FLOOR` | `0.40` | launch_app accept bar when a strict `.desktop` hit corroborates |
| `HEARD_DICT_CUTOFF` | `90` | rapidfuzz score for that dictionary hit |

Tune thresholds from benchmark output, not feel: misaccepts are the hard gate.

---

## Repo structure

```
heard/
├── pyproject.toml               # entry point: uv run heard listen
├── .github/workflows/test.yml   # CI: ruff + mypy + pytest on push/PR
├── heard/
│   ├── cli.py                   # warmup, preflight, loop, hold bindings, stage timings
│   ├── stt.py                   # capture bus, streaming hold STT, VAD trim
│   ├── intent.py                # fast path first, generative fallback, checkpoint resolution
│   ├── classifier.py            # bilingual prototypes, centroids, gating, slot extraction
│   ├── embedder.py              # needle-contrastive / fastembed backends
│   ├── llm_query.py             # question/pregunta branch: LLM answer + TTS chain
│   ├── config.py                # config.toml load/save
│   ├── ipc.py                   # unix-socket hold server + one-shot clients
│   ├── lock.py                  # single-instance flock guard
│   ├── events.py                # local JSONL usage log
│   ├── shell_allowlist.py       # safe command templates
│   └── tools/
│       ├── registry.py          # name -> handler dispatch + validate
│       ├── types.py             # Ok, Rejected, Failed, ParamSpec, Entry
│       ├── apps.py              # focus-or-launch over the WM backend
│       ├── media.py              # MPRIS over jeepney, playerctl fallback
│       ├── volume.py            # wpctl
│       ├── microphone.py        # mic mute/unmute via wpctl
│       ├── screen_record.py     # wf-recorder / ffmpeg x11grab, pidfile state
│       ├── window.py            # close/focus/fullscreen via WM backend
│       ├── workspace.py         # workspace switch via WM backend
│       ├── system_query.py      # battery, time, network, disk
│       └── helpers/
│           ├── wm.py            # Hyprland / Sway / KDE / GNOME backends (GNOME: extension first, Eval fallback)
│           ├── apps.py          # .desktop enumeration (all locales) + rapidfuzz
│           ├── network.py       # connectivity checks
│           └── system.py        # format_bytes etc.
├── packaging/systemd/heard.service
├── packaging/gnome-shell/heard@heard/  # companion Shell extension: D-Bus window tools
├── tests/                       # 299 tests across 26 files
├── checkpoints/                 # gitignored: needle weights
├── scripts/
│   ├── benchmark_latency.py     # per-stage probes: --only intent|stt|dispatch|needle [--audio clip.wav]
│   └── export_fallback.py       # fallback-speed spike: step profile + ONNX attempt
└── reports/
    ├── latency.md               # measured numbers + threshold tuning tables
    └── fallback_speed.md        # generative-tail investigation notes
```

---

## Testing & benchmarking

```bash
uv run pytest                                            # 299 tests
uv run python scripts/benchmark_latency.py --only intent # fast vs generative + tail split
uv run python scripts/benchmark_latency.py --only stt    # audio finalize + whisper
uv run python scripts/benchmark_latency.py --only dispatch
uv run python scripts/benchmark_latency.py --audio cmd.wav
```

---

## Acceptance criteria (v1.0)

- ✅ Each of the 6 tools resolves correctly from casual phrasing, EN and ES (probe: 19/20 EN + 16/16 ES fast-path, 0 misaccepts).
- ✅ Classifier-accepted commands resolve intent in <100ms post-STT.
- ✅ Daemon mode: systemd unit, single-instance lock, unix-socket hold bindings for Hyprland/Sway/KDE.
- ✅ Local JSONL event log for threshold tuning.
- ⏳ End-to-end key-release → feedback under 1s on i5-1335U: STT tail numbers being collected from live `[...]` lines; fill [`reports/latency.md`](reports/latency.md).
- ✅ Generative Needle remains fallback-only (fires on 1-3 of 20 typical commands).
- ✅ Unresolvable input prints a declined/unparseable message; it never crashes.

## Known limitations

- Bilingual EN/ES only; more languages need prototypes + a multilingual embedder entry (mechanism exists).
- The bundled Needle checkpoint has an untrained contrastive head: the fast path therefore depends on fastembed (one-time model download). A retrieval-finetuned checkpoint would flip `HEARD_EMBEDDER=needle` back on.
- Generative fallback costs seconds (fp32 JAX decode); fine at 5-15% traffic, painful above it.
- GNOME needs the companion Shell extension for full support; without it only unsafe-mode shells work (Eval fallback).

---

## Roadmap (post-1.0)

| Area | What lands |
|---|---|
| **Responder** | Floating GTK popup + local TTS (piper/espeak-ng) over the existing unix socket |
| **Fallback speed** | KV-cache + ONNX spike: [`reports/fallback_speed.md`](reports/fallback_speed.md), run via `scripts/export_fallback.py` |
| **Languages** | More languages via the existing prototype + multilingual-embedder mechanism |
| **Analytics** | Daily rollups over the JSONL log; retrain pipeline (failures → finetune data) |

The model stays 26M Needle at the core. Later releases add feedback surfaces, a faster fallback tail, and broader language coverage.
