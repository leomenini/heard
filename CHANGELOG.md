# Changelog

## 1.0.0 — 2026-08-22

First complete release: capture → resolve → dispatch runs end-to-end on
consumer hardware with sub-second perceived latency for classifier-accepted
commands, in English or Spanish.

**Fast path**
- Embedding nearest-centroid classifier over ~150 bilingual prototypes
  (17 tool-action classes + explicit unknown centroid for rejection)
- Finetuned Needle contrastive head (`scripts/train_needle_head.py`) so
  `HEARD_EMBEDDER=needle` runs the fast path without the fastembed ONNX
  dependency; JIT'd fixed-shape encoder + backend-specific threshold
- Slot resolution by lookup, not generation: spoken numbers EN/ES,
  percent phrases, `.desktop` fuzzy app matching (all locales indexed)
- Dictionary-corroborated launch gate; cosine thresholds env-tunable
- Generative Needle decode demoted to fallback (fires on ~5–15% of traffic)

**STT**
- Persistent PortAudio stream into an offset-addressed ring buffer
- Streaming during hold: pause-delimited segments finalize in a background
  thread; key release only transcribes the post-pause tail
- Energy VAD trim, fixed language, beam 1, int8

**Daemon & bindings**
- `heard listen` serves `$XDG_RUNTIME_DIR/heard.sock`
- `heard hold down` / `heard hold up` one-shot clients for compositor
  keybindings (Hyprland / Sway / KDE)
- systemd user unit in `packaging/systemd/`
- Single-instance lock via flock'd pidfile

**Tools**
- WM abstraction: Hyprland + Sway + KDE (kdotool windows, KWin-script workspaces)
- focus-or-launch semantics in `launch_app`
- `media_control` over persistent MPRIS D-Bus (jeepney), playerctl fallback

**Platform**
- Config file (`~/.config/heard/config.toml`) + `heard config get/set/show`
- Local JSONL event log (opt-out) feeding threshold tuning
- Per-stage timing lines on every command
- Bilingual EN/ES end to end
- 234 tests; ruff clean; CI runs lint + tests

## 0.1.0 — 2026-07

Initial skeleton: evdev push-to-talk, blocking Whisper transcription,
generative-only Needle intent, six Hyprland-bound tools.
