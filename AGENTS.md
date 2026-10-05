# AGENTS.md — working context for AI assistants on this fork

Local notes, deliberately untracked (same family as CLAUDE.md / TLDR.md /
PLAN-X11.md). Read PLAN-X11.md next to this file for raw evidence numbers
and the full backlog. Updated 2026-08-25 after the X11/Cinnamon milestone
session.

## Repository relationship

- This checkout: `/home/leo/devTools/Heard/heard`
- Fork: `origin` = leomenini/heard (push via SSH only, see below)
- Upstream: `upstream` = emiliano-go/heard (public, MIT, solo maintainer,
  default branch `master`, this was its first-ever external PR)
- Local `master` is 5 commits ahead of origin/master, nothing force-pushed,
  master itself never pushed this session.

## Current state

| ref | where | contents |
|---|---|---|
| `master` (local) | ahead of origin by 5 | upstream sync (`cba2ac0` LICENSE) + 4 thematic commits |
| `fix/weightless-degradation` | **pushed to origin** | cherry-pick of commit A alone on top of upstream/master (`0c0f13b`) |
| PR #1 | **DRAFT on upstream** | https://github.com/emiliano-go/heard/pull/1 — commit A, awaiting Leo's review, then "Ready for review" |

The four local thematic commits:

1. `6120e34` feat: generic X11/EWMH backend via wmctrl (Cinnamon/XFCE/MATE/i3)
2. `3a04923` fix: degrade gracefully when the Needle checkpoint is missing
   (= PR #1, also cherry-picked as `0c0f13b` on its branch)
3. `702eaf0` feat: opt-in dictionary rescue behind HEARD_DICT_RESCUE
4. `587cd57` feat(window): minimize action on x11 (ICCCM WM_CHANGE_STATE via
   python-xlib; Muffin ignores wmctrl's EWMH hidden state)

## Why commit A matters (the elevator pitch)

Upstream documented running without weights but two paths crashed anyway:
(1) `_encoder()` passed `needle_triple=_model()` — eager argument evaluation
ran `_model()` at warmup and raised on missing checkpoint, killing startup;
(2) `_generate_resolve()` called `_model()` bare, so one ambiguous utterance
killed the whole daemon. A makes both degrade: lazy factory + printed reason,
decline instead of crash. Byte-identical behavior when weights exist. Zero
design opinions: follows house style (print "heard: ...", flush=True) and
house contracts (handlers return Ok|Rejected|Failed, never raise).

## PR strategy (small-first; solo maintainer, first external PR ever)

| slice | commit | size | controversy | status |
|---|---|---|---|---|
| A weightless fix | `3a04923` | 3 files +64/-5 | none (bugfix, conforms to conventions) | **draft PR #1 open** |
| B x11 backend | `6120e34` | ~310 lines | low | waiting on A |
| C minimize x11 | `587cd57` | ~270 lines | medium (new python-xlib runtime dep) | depends on B |
| D dict rescue | `702eaf0` | ~140 lines | high (classifier gating semantics) | maybe keep fork-local until maintainer signals appetite |

Rules agreed for PR bodies: lead with how the change follows existing
conventions before introducing anything new; never preemptively apologize
for choices that are house style; include repro, behavior-unchanged-when
statement, test counts, live verification note.

## Machine facts (Mint 22 / Cinnamon / Muffin / X11 / PipeWire)

- AMD A10-9700 4c/4t AVX2 (no VNNI): embed ~190ms, STT tail 1.6-2.8s,
  intent 74-166ms, dispatch <20ms, whisper offline decode ~0.5x realtime.
- No checkpoints dir -> weightless mode: fastembed multilingual MiniLM,
  generative fallback declines, FALLBACK_CLASSES ({window_action:focus})
  always declines.
- Tooling present: wmctrl, xprop, xwininfo, wpctl, pactl, ffmpeg. Absent:
  xdotool, wf-recorder, uwsm, playerctl, kdotool, xterm.
- GT 1030 2GB sits idle (heard stack is CPU-only).

## Shell traps hit this session (do not repeat)

- `pkill -f "heard listen"` matches its own wrapping `bash -c` string and
  hangs/kills the tool shell. Use `pkill -f "heard [l]isten"` or pgrep+kill.
- Long-running daemons must be launched detached:
  `setsid bash -c '...' < /dev/null & disown`; plain `nohup ... &` dies with
  the tool timeout.
- HTTPS push from these shells fails (no interactive credential prompt).
  Push works via explicit SSH url:
  `git push git@github.com:leomenini/<repo>.git <branch>` (SSH auth OK).
- Visible terminal for the daemon:
  `gnome-terminal --working-directory=<repo> -- /bin/bash -c 'uv run heard listen; echo "--- daemon exited ---"; exec bash'`
  Warmup ~15s cached. Config: ~/.config/heard/config.toml (language=en
  currently; en decodes far better than es on whisper base).
- Scratch windows for WM tests: gnome-terminal --title=HEARD-MIN-SCRATCH.
  Never voice-test close/fullscreen/minimize without focusing scratch first.
- Mic restore if a test mutes it: wpctl set-source-mute @DEFAULT_SOURCE@ 0

## Conventions enforced this session

- All repo artifacts in English (code, comments, docs, commits, PRs). Chat
  language follows the user's lead.
- No Co-Authored-By / Claude trailers in commits (user global preference).
- Thematic single-purpose commits; CHANGELOG under "## Unreleased";
  benchmark numbers go in reports/, never commit messages.
- Prose in README/CHANGELOG uses no em dashes.
- Lazy heavy imports stay lazy (PLC0415); every ruff ignore carries a
  justification comment; handlers never raise to caller.
- User approval before ANY code change beyond the agreed plan; drop
  unauthorized edits on request (happened once with extra stt tests).

## Known issues to disclose upstream eventually

- tests/test_stt_helpers.py::TestTranscribeSignature reads the real
  ~/.config/heard/config.toml (not hermetic). Surfaces only when a dev
  config exists; currently passes because language=en. Unfixed on purpose
  this session.
- minimize is x11-only; other backends raise unsupported honestly.
- python-xlib is a new runtime dep justified solely by Muffin refusing the
  EWMH hidden state (verified by xprop before implementing).

## Open items

1. Watch PR #1 CI; mark ready after Leo reviews on GitHub.
2. reports/x11_cinnamon.md consolidated evidence report (raw material lives
   in PLAN-X11.md).
3. Docker verify image (docker/Dockerfile + docker/verify.sh; ubuntu:24.04
   base, Xvfb+openbox+pulseaudio matrix; check_ptt blocks live PTT in
   container - document that limit like jarvis README does).
4. Live HEARD_DICT_RESCUE=1 daemon pass (only inline benchmarks exist).
5. Re-check upstream drift before each PR: git fetch upstream.
6. Optional A/Bs: stt_cpu_threads=2; stt_model_size=small (~3x tail cost).

## Backlog (details in PLAN-X11.md)

Mine events.jsonl phrasings into PROTOTYPES -> dict-rescue default decision
-> contrastive-head finetune (flips HEARD_EMBEDDER=needle) -> LoRA fallback
finetune (needle doc/finetuning.md; VERIFY .pkl vs .cact loader against
pinned rev ffb1c51 first) -> GNOME/KWin minimize branches -> optional GPU
stt_device keys. Jarvis cross-pollination deferred: tile halves, post-launch
placement, piper for jarvis `say`; jarvis v2-tiling branch unmerged + chmod
pending there.
