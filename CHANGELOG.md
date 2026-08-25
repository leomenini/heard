# Changelog

## Unreleased

**Generic X11 backend**
- Fifth WM backend (`x11`) speaks plain EWMH through `wmctrl`: window list,
  close, fullscreen, focus, and workspace switch. Detection tries it last, so
  Cinnamon, XFCE, MATE, and i3 sessions get support without displacing the
  richer Hyprland/Sway/KDE/GNOME backends.
- `launch_app` falls back to a plain spawn where `uwsm` is absent (X11
  desktops) instead of declining.
- Screen recording grabs `$DISPLAY` instead of assuming `:0.0`.
- `window_action` gains `minimize`, implemented for the x11 backend via an
  ICCCM `WM_CHANGE_STATE` client message (python-xlib): Muffin ignores the
  EWMH hidden state that wmctrl sends, so the taskbar-style request is the
  one that actually iconifies on Cinnamon. Other backends decline honestly
  rather than mapping minimize onto close.

**Weightless degradation**
- A missing Needle checkpoint no longer kills warmup or mid-command dispatch:
  `resolve_encoder` takes a lazy factory, `auto` degrades to fastembed, and an
  utterance routed to the generative fallback declines with a printed reason
  instead of raising.

**Opt-in dictionary rescue**
- `HEARD_DICT_RESCUE=1` lets a strict `.desktop` hit rescue a near-miss
  `launch_app` that placed within `HEARD_RESCUE_RANK` (default 3) above
  `HEARD_RESCUE_FLOOR` (default 0.30). Off by default: it trades accept rate
  against misaccept risk; measured numbers live in `reports/latency.md`.

## 1.4.0 — 2026-08-22

**GNOME without unsafe mode**
- Companion Shell extension (`packaging/gnome-shell/heard@heard`) exposes
  window list, focus, close, fullscreen, and workspace switch over a
  `dev.heard.WindowTools` D-Bus service — no unsafe mode required.
- Two-tier transport in the WM abstraction: extension first (absence is
  latched for zero steady-state cost), `Eval` fallback kept for unsafe-mode
  shells; window ids unified on `get_stable_sequence`.
- Install guide + manual QA checklist in `packaging/gnome-shell/README.md`.

**KDE fullscreen**
- Fullscreen toggle now supported on Plasma via a one-shot KWin script with
  Plasma 5 / 6 runtime detection.

**Local-first TTS**
- `piper-tts` is the primary speech backend for query answers: local ONNX,
  voice models downloaded once per language into `~/.cache/heard/piper`.
- `gTTS` moved to the optional `[tts]` extra; auto chain is now
  piper → gtts → flite → console.
- New config key `tts_backend`: `""`(auto) | `piper` | `gtts` | `flite` |
  `none`; README documents exactly what can touch the network.

**Platform**
- mypy adopted (`check_untyped_defs`, targeted import overrides); CI now
  runs ruff + mypy + pytest; 10 type findings fixed incl. an IPC stop race.
- Consistency pass: real package description, README version/test-count
  drift fixed, public names for network helpers, relative imports in
  `system_query`, `HoldController.is_busy()`/`cancel_pending()` replace
  private-attribute pokes.
- Fallback-speed spike: `scripts/export_fallback.py` + 
  `reports/fallback_speed.md` — decoder re-forwards its full prefix every
  token (no KV cache); KV-caching identified as the primary fix.
- 299 tests.

## 1.3.0 — 2026-08-22

**GNOME support**
- Added `gnome` to the window-manager abstraction.
- Detects GNOME via `XDG_CURRENT_DESKTOP` / `DESKTOP_SESSION` or config
  `wm_backend = "gnome"`.
- Window listing, focus, close active, fullscreen toggle, and workspace switch
  are implemented through GNOME Shell's D-Bus `Eval` interface.

## 1.2.0 — 2026-08-22

**Screen recording**
- New `screen_record` tool: `screenrecord start` / `screenrecord stop`.
- Records the default display or a named one (`screenrecord start One`).
- Display labels are mapped to output names via `screenrecord_outputs`
  (e.g. `One=DP-1,Two=HDMI-1`); `screenrecord_output` sets the default.
- Recordings are saved to `screenrecord_folder` (default `~/Videos`).
- Uses `wf-recorder` on Wayland, falls back to `ffmpeg -f x11grab` on X11.
- State is tracked in a pidfile so `stop` finalises the file.

## 1.1.0 — 2026-08-22

**Query mode**
- Transcripts starting with "Question" or "Pregunta" are sent to a configured
  OpenAI-compatible LLM (`llm_url`, `llm_api_key`, `llm_model`) and the answer
  is spoken out loud via TTS (`language` and `tts_enabled` config keys).
- TTS uses `gTTS` by default (en/es), with local `flite` fallback for English
  and console fallback if no player is available; playback runs in a background
  thread so the daemon stays responsive.

**Microphone control**
- New `microphone_control` tool: `mute` / `unmute` the default microphone source
  via `wpctl @DEFAULT_AUDIO_SOURCE@`.

**Config**
- New keys: `llm_url`, `llm_api_key`, `llm_model`, `tts_enabled`.

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
