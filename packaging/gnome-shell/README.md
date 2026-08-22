# heard window-tools Shell extension

GNOME Shell restricts `org.gnome.Shell.Eval` to unsafe mode (GNOME 41+), so
`heard` ships this tiny companion extension that exposes the few window
operations the daemon needs as a session-bus service:

    bus/path/interface:  dev.heard.WindowTools  /dev/heard/WindowTools
    methods: ListWindows() -> s(json) | Activate(id i) | Close("active"|id s)
             ToggleFullscreen() | SetWorkspace(n i)

heard tries this service first and falls back to Eval automatically (which
keeps working for users who already run unsafe mode).

## Install

```bash
mkdir -p ~/.local/share/gnome-shell/extensions
cp -r packaging/gnome-shell/heard@heard ~/.local/share/gnome-shell/extensions/
gnome-extensions enable heard@heard
# X11 sessions: Alt+F2, r, Enter. Wayland: log out and back in.
```

Requires GNOME 45+. After enabling (or disabling) the extension, restart
`heard listen` so its one-shot probe re-runs.

## Manual QA checklist

Run on a real GNOME session (Wayland), then from a terminal:

```bash
gdbus call --session --dest dev.heard.WindowTools \
    --object-path /dev/heard/WindowTools \
    --method dev.heard.WindowTools.ListWindows
```

- [ ] `ListWindows` returns JSON of open windows (native Wayland + XWayland)
- [ ] `Activate` with a listed id focuses that window
- [ ] `Close "active"` closes the focused window; `Close <id>` closes by id
- [ ] `ToggleFullscreen` toggles the focused window
- [ ] `SetWorkspace 2` switches to workspace 2 (1-based)
- [ ] With extension disabled: heard commands still work when shell runs in
      unsafe mode (Eval fallback), and fail cleanly otherwise
- [ ] `journalctl --user -u heard -f` shows no repeated 1s stalls once the
      absence latch has flipped (extension missing)
