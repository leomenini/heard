# PLAN-X11 — Cinnamon/X11 milestone status and next steps

Local working notes (untracked on purpose). Fork: leomenini/heard,
upstream: emiliano-go/heard. Updated 2026-08-25.

## Landed (local master, nothing pushed)

| commit | what it is | proof it works |
|---|---|---|
| `6120e34` feat: generic X11/EWMH backend via wmctrl | fifth WM backend tried last; launch_app plain-spawn fallback; `$DISPLAY` in screen_record | 12 x11 tests; live session matches test fixtures (Muffin, 4 workspaces); ruff+mypy+326 tests green at commit time |
| `3a04923` fix: degrade gracefully when checkpoint missing | lazy needle factory; fallback declines instead of killing daemon | verified live: startup prints "needle checkpoint unavailable ... using fastembed", mid-command decline prints reason |
| `702eaf0` feat: opt-in dictionary rescue | HEARD_DICT_RESCUE / RANK=3 / FLOOR=0.30; own floor so tuning cannot loosen corroboration gate | held-out probes: 9/12 accepts, 0 misaccepts at defaults (fastembed geometry) |
| `587cd57` feat(window): minimize action on x11 | ICCCM WM_CHANGE_STATE via python-xlib; registry enum + centroid; honest unsupported elsewhere | live Muffin iconify verified (WM_STATE.state=3); geometry gate below |

Base = upstream master (`cba2ac0 Create LICENSE`). Sync via:
`git fetch upstream && git merge --ff-only upstream/master`.

## Session evidence (2026-08-25, Mint 22 / Cinnamon / X11 / PipeWire)

Hardware: AMD A10-9700 4c/4t (AVX2, no AVX512/VNNI), 10GB RAM, GT 1030 idle
(heard's stack is CPU-only today).

Latency vs README reference (i5-1335U):

| stage | reference | this machine |
|---|---|---|
| embed per command | 53ms | ~190ms |
| intent total (live logs) | <100ms | 74–166ms |
| STT tail | ~240ms | ~1.6–2.8s |
| dispatch | <20ms | <20ms |
| warmup ready | 21s | ~15s (weights cached); first fastembed load 32s incl download |

Whisper base int8 offline decode: 8s audio -> ~4.2s.

Muffin quirk (the reason python-xlib exists here): `wmctrl -r :ACTIVE: -b
add,hidden` is ignored (WM_STATE stays Normal); the ICCCM
`WM_CHANGE_STATE=IconicState` client message iconifies reliably. Verified by
xprop before writing any code. xdotool is absent on stock Mint; kdotool
pattern covers KDE separately.

Misaccept story that motivated minimize:

```
before: "Minimizar ventana actual." -> window_action:close  (fast 0.69) CLOSED the window
after:  same phrase                 -> window_action:minimize (0.935)
        close traffic untouched (0.814–0.956); off-topic misaccepts 0
```

STT notes: whisper base/es garbles casual/distant speech but fails SAFE
(garbage -> below floor -> generative -> weightless decline). Clean English
decodes best under `language=en`. Doubled transcript once observed was user
repetition, not a pipeline bug: holds under HOLD_STREAM_THRESHOLD_S use the
same single-piece path as offline transcription, and segment accounting is
exact (pieces partition hold audio). Root cause of "nothing happens" feeling:
no feedback surface exists yet (roadmap Responder); daemon prints only to its
stdout.

Weightless behavior confirmed live: FALLBACK_CLASSES = {window_action:focus}
declines with printed reason; every other action in the registry fast-paths.
Event log (~/.local/state/heard/events.jsonl) captured full command history
with scores — doubles as future finetune data.

## Open items before PR revision

1. `reports/x11_cinnamon.md` — the consolidated evidence report (this file is
   raw material for it).
2. Docker verification image: docker/Dockerfile + docker/verify.sh,
   ubuntu:24.04 (= Mint 22 baseline) + uv + CPython 3.13 + Xvfb/openbox/
   wmctrl/x11-utils/xterm/ffmpeg/pulseaudio, non-root user, clean clone;
   matrix: detect->x11, wm ops on scratch windows, focus-or-launch,
   .desktop enumeration, screen_record mp4, STT from synthesized WAV,
   weightless declines, full pytest inside. Named cache volume for weights.
   Known limit: check_ptt exits hard without /dev/input, so no live PTT in
   container (same caveat as jarvis README).
3. Live HEARD_DICT_RESCUE=1 pass through the real daemon (benchmarked inline
   only so far).
4. `git fetch upstream` drift re-check immediately before PR.
5. Optional A/Bs: stt_cpu_threads=2 vs default; stt_model_size=small
   (accuracy up, tail cost ~3x on this CPU).

## Known issues to disclose in the PR

- tests/test_stt_helpers.py::TestTranscribeSignature reads the developer's
  real ~/.config/heard/config.toml (pre-existing hermeticity bug; surfaces
  when a config exists; CI never sees it). Left unfixed deliberately this
  session.
- minimize is x11-only by design; hyprland/sway/kde/gnome raise unsupported
  rather than mapping onto close.
- python-xlib is a new runtime dependency (pure-python, lazily imported);
  justification: Muffin refuses the EWMH path and xdotool is not installed
  by default on Mint.
- Cinnamon custom shortcuts fire on press only (no press/release pair), so
  socket-driven holds are limited there; evdev PTT is the primary path
  (README already documents this).

## Backlog ladder (checkpoint work)

0. Mine events.jsonl phrasings into classifier PROTOTYPES (minutes, no ML).
1. Decide dict-rescue default-on after more machines benchmark it.
2. Contrastive-head finetune (upstream has machinery) -> flips
   HEARD_EMBEDDER=needle back on; attacks app-name ranking at the root.
3. Full LoRA on generative fallback (needle doc/finetuning.md): few hundred
   examples move tool selection; thousands needed for argument grounding;
   include answers:[] off-topic rows; CPU-feasible for small sets.
   OPEN TECHNICALITY: heard loads checkpoints/*.pkl, current needle exports
   .cact — verify loader compatibility against pinned rev ffb1c51 first.
4. GNOME extension + KWin branches for minimize.
5. GPU follow-up: stt_device/stt_compute_type config keys (GT 1030 2GB sits
   idle; CTranslate2 could offload whisper).

## jarvisAssistant cross-pollination (evaluated, deferred)

- Tile halves: _work_area parses WA: from wmctrl -d (not xrandr),
  _NET_FRAME_EXTENTS via xprop subtracted from client-area geometry,
  unmaximize before resize. Would become new enumerable window_action values
  -> needs prototypes + held-out probes + misaccept counting per CLAUDE.md.
- Post-launch placement: wait-for-new-window polling then workspace/tile.
- Reverse: implement jarvis's reserved `say` action using heard's piper chain
  (SPEC section 8). Housekeeping: v2-tiling branch unmerged to main; chmod
  pending on jarvis.py.
