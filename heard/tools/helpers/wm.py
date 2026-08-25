"""Window-manager abstraction: Hyprland, Sway, KDE, and GNOME.

Detection order: config `wm_backend` override, then environment
(HYPRLAND_INSTANCE_SIGNATURE / SWAYSOCK / KDE / GNOME markers). Every command
raises on failure -- handlers translate that into Failed/Rejected results.

GNOME is tiered: the companion Shell extension (packaging/gnome-shell) is
tried first via its dev.heard.WindowTools D-Bus service; if it is absent we
fall back to org.gnome.Shell.Eval, which only works when the shell runs in
unsafe mode (GNOME 41+).
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

from jeepney import DBusAddress, new_method_call

BACKENDS = ("hyprland", "sway", "kde", "gnome", "x11")
_DBUS_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class Window:
    address: str   # backend-specific handle usable for focusing
    cls: str       # app class / app_id / resourceClass
    title: str

    def label(self) -> str:
        return f"{self.cls} {self.title}".lower().strip()


def _configured_backend() -> str:
    try:
        from ... import config

        return str(config.cached("wm_backend") or "auto").lower()
    except Exception:
        return "auto"


def detect() -> str | None:
    override = _configured_backend()
    if override in BACKENDS:
        return override
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return "hyprland"
    if os.environ.get("SWAYSOCK"):
        return "sway"
    if (os.environ.get("KDE_FULL_SESSION", "").lower() == "true"
            or "kde" in os.environ.get("XDG_CURRENT_DESKTOP", "").lower()):
        return "kde"
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    session = os.environ.get("DESKTOP_SESSION", "").lower()
    if "gnome" in desktop or "gnome" in session:
        return "gnome"
    # Generic EWMH last: any X11 session without a richer backend above
    # (Cinnamon, XFCE, MATE, i3 ...). Kept last so KDE-on-X11 and GNOME-on-X11
    # still get their own backends.
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "x11":
        return "x11"
    if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return "x11"
    return None


def require() -> str:
    backend = detect()
    if backend is None:
        raise RuntimeError(
            "no supported window manager detected "
            "(tried Hyprland, Sway, KDE, GNOME); set wm_backend in config to override"
        )
    return backend


def _run(cmd: list[str], timeout: float = 3.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          check=True, timeout=timeout)


# --- Hyprland --------------------------------------------------------------

def _hyprland_list() -> list[Window]:
    out = _run(["hyprctl", "clients", "-j"]).stdout
    clients = json.loads(out) if out.strip() else []
    windows = []
    for c in clients if isinstance(clients, list) else []:
        addr = c.get("address")
        if not addr:
            continue
        cls = c.get("initialClass") or c.get("class") or ""
        title = c.get("title") or ""
        windows.append(Window(addr, str(cls), str(title)))
    return windows


# --- Sway ------------------------------------------------------------------

def _sway_walk(node: dict, out: list[Window]) -> None:
    if (node.get("type") in ("con", "floating_con")
            and node.get("name")
            and (node.get("app_id") or node.get("window_properties"))):
        props = node.get("window_properties") or {}
        cls = node.get("app_id") or props.get("class") or ""
        out.append(Window(str(node["id"]), str(cls), str(node["name"])))
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        _sway_walk(child, out)


def _sway_list() -> list[Window]:
    out = _run(["swaymsg", "-t", "get_tree", "--raw"]).stdout
    tree = json.loads(out) if out.strip() else {}
    windows: list[Window] = []
    if isinstance(tree, dict):
        _sway_walk(tree, windows)
    return windows


# --- GNOME -----------------------------------------------------------------

_GNOME_SHELL = DBusAddress("/org/gnome/Shell", bus_name="org.gnome.Shell",
                           interface="org.gnome.Shell")
_GNOME_SERVICE = DBusAddress("/dev/heard/WindowTools",
                             bus_name="dev.heard.WindowTools",
                             interface="dev.heard.WindowTools")
_GNOME_TIMEOUT_S = 1.0

_gnome_extension_dead = False   # latched when the extension is found absent


def _dbus_session_connection():
    """Session bus, reusing the MPRIS module's persistent connection."""
    from jeepney.io.blocking import open_dbus_connection

    try:
        from .. import media

        return media._connection()
    except Exception:
        return open_dbus_connection()


def _gnome_call(method: str, signature: str | None = None,
                body=None) -> str:
    """Call the companion Shell extension; return its string out-arg.

    Raises whenever the extension is not on the bus so callers fall back to
    Eval. The first failure latches for the daemon lifetime -- steady state
    then costs no timeout at all. Restart heard after enabling the extension.
    """
    global _gnome_extension_dead
    if _gnome_extension_dead:
        raise RuntimeError("heard window-tools extension unavailable")
    conn = _dbus_session_connection()
    reply = conn.send_message(new_method_call(_GNOME_SERVICE, method,
                                              signature, body),
                              timeout=_GNOME_TIMEOUT_S)
    if reply is None or reply.message_type == 4:  # error
        _gnome_extension_dead = True
        raise RuntimeError("heard window-tools extension call failed")
    return str(reply.body[0]) if reply.body else ""


def _gnome_eval(js: str) -> str:
    """Run JS inside GNOME Shell via D-Bus Eval (unsafe-mode fallback)."""
    conn = _dbus_session_connection()
    reply = conn.send_message(new_method_call(_GNOME_SHELL, "Eval", "s", (js,)),
                              timeout=_DBUS_TIMEOUT_S)
    if reply is None or reply.message_type == 4:  # error
        raise RuntimeError("gnome shell eval failed")
    success, result = reply.body
    if not success:
        raise RuntimeError(f"gnome shell eval error: {result}")
    return result


# Eval snippets report ids like the extension does: stable_sequence where
# available, get_id() on shells that lack it.
_ID_EXPR = ("(typeof w.get_stable_sequence === 'function' "
            "? w.get_stable_sequence() : w.get_id())")

_GNOME_LIST_JS = (
    "JSON.stringify(global.get_window_actors().map(a => {"
    "let w = a.meta_window;"
    f"return {{id: {_ID_EXPR},"
    "cls: w.get_wm_class() || '',"
    "title: w.get_title() || ''}};}))"
)


def _windows_from_json(raw: str) -> list[Window]:
    """Parse the shared [{id, cls, title}] payload used by both GNOME tiers."""
    windows = []
    for item in json.loads(raw) if raw and raw.strip() else []:
        windows.append(Window(str(item["id"]), str(item.get("cls", "")),
                              str(item.get("title", ""))))
    return windows


def _gnome_list() -> list[Window]:
    try:
        raw = _gnome_call("ListWindows")
    except Exception:
        raw = _gnome_eval(_GNOME_LIST_JS)
    return _windows_from_json(raw)


# --- KDE -------------------------------------------------------------------

_KWIN_SCRIPTING = DBusAddress("/KWin", bus_name="org.kde.KWin",
                              interface="org.kde.KWin.Scripting")


def _dbus_call(address: DBusAddress, method: str, signature=None, body=None):
    conn = _dbus_session_connection()
    reply = conn.send_message(new_method_call(address, method, signature, body),
                              timeout=_DBUS_TIMEOUT_S)
    if reply is None or reply.message_type == 4:  # error
        raise RuntimeError(f"kwin dbus call {method} failed")
    return reply


def _kde_kdotool(*args: str) -> str:
    return _run(["kdotool", *args]).stdout.strip()


def _kde_list() -> list[Window]:
    ids = [i for i in _kde_kdotool("search", ".").splitlines() if i.strip()]
    windows = []
    for wid in ids[:200]:                       # sanity cap
        try:
            title = _kde_kdotool("getwindowname", wid)
            cls = _kde_kdotool("getwindowclassname", wid)
        except subprocess.CalledProcessError:
            continue                            # window vanished mid-listing
        windows.append(Window(wid, cls, title))
    return windows


def _kwin_run_script(source: str) -> None:
    """One-shot KWin script over D-Bus (works on Plasma 5 and 6)."""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        reply = _dbus_call(_KWIN_SCRIPTING, "loadScript", "s", (path,))
        sid = int(reply.body[0])
        script = DBusAddress(f"/Scripting/Script{sid}", bus_name="org.kde.KWin",
                             interface="org.kde.KWin.Script")
        _dbus_call(script, "run")
        _dbus_call(script, "stop")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# --- X11 (generic EWMH via wmctrl) -----------------------------------------

_X11_ACTIVE = ":ACTIVE:"       # wmctrl's own selector; saves an xprop round trip


def _x11_window(line: str) -> Window | None:
    """Parse one `wmctrl -l -x` row: id desktop wm_class host [title].

    Title is the only field that may contain spaces, so it takes the tail.
    A window with an empty title is still a window; a short row is not.
    """
    parts = line.split(None, 4)
    if len(parts) < 4:
        return None
    wid, _desktop, wm_class, _host = parts[:4]
    title = parts[4] if len(parts) > 4 else ""
    # WM_CLASS is 'instance.Class'; the class half matches what the other
    # backends report (hyprland initialClass, sway app_id)
    cls = wm_class.rsplit(".", 1)[-1] if "." in wm_class else wm_class
    return Window(wid, cls, title)


def _x11_list() -> list[Window]:
    out = _run(["wmctrl", "-l", "-x"]).stdout
    windows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        window = _x11_window(line)
        if window is not None:
            windows.append(window)
    return windows


def _x11_minimize() -> None:
    """Iconify the active window via the ICCCM WM_CHANGE_STATE message.

    Muffin (Cinnamon) ignores the EWMH _NET_WM_STATE_HIDDEN client message
    that `wmctrl -b add,hidden` sends, but honors WM_CHANGE_STATE=Iconic --
    the same request taskbars and xdotool send. python-xlib is imported
    lazily; a missing install surfaces as RuntimeError so the tool handler
    reports Failed instead of raising into dispatch.
    """
    try:
        from Xlib import X
        from Xlib import display as xdisplay
        from Xlib.protocol import event as xevent
    except ImportError as e:
        raise RuntimeError(f"python-xlib is required for minimize on x11: {e}") from e

    d = xdisplay.Display()
    try:
        root = d.screen().root
        prop = root.get_full_property(d.intern_atom("_NET_ACTIVE_WINDOW"),
                                      X.AnyPropertyType)
        wid = int(prop.value[0]) if prop is not None and len(prop.value) else 0
        if not wid:
            raise RuntimeError("no active window to minimize")
        msg = xevent.ClientMessage(
            window=wid,
            client_type=d.intern_atom("WM_CHANGE_STATE"),
            sequence_number=0,
            data=(32, [3, 0, 0, 0, 0]),       # 3 == IconicState
        )
        root.send_event(msg, event_mask=X.SubstructureRedirectMask
                        | X.SubstructureNotifyMask)
        d.flush()
    finally:
        d.close()


# --- Public API ------------------------------------------------------------

def _unsupported(backend: str) -> RuntimeError:
    return RuntimeError(f"unsupported wm backend {backend!r}")


def list_windows() -> list[Window]:
    backend = require()
    if backend == "hyprland":
        return _hyprland_list()
    if backend == "sway":
        return _sway_list()
    if backend == "gnome":
        return _gnome_list()
    if backend == "x11":
        return _x11_list()
    if backend == "kde":
        return _kde_list()
    raise _unsupported(backend)


def close_active() -> None:
    backend = require()
    if backend == "hyprland":
        _run(["hyprctl", "dispatch", "closewindow", "active"])
    elif backend == "sway":
        _run(["swaymsg", "kill"])
    elif backend == "gnome":
        try:
            _gnome_call("Close", "s", ("active",))
        except Exception:
            _gnome_eval("global.display.focus_window.delete(0); 'done'")
    elif backend == "x11":
        _run(["wmctrl", "-c", _X11_ACTIVE])
    elif backend == "kde":
        active = _kde_kdotool("getactivewindow")
        _kde_kdotool("windowclose", active)
    else:
        raise _unsupported(backend)


_KDE_FULLSCREEN_SCRIPT = (
    "if (workspace.activeWindow) {"                  # Plasma 6
    " workspace.activeWindow.fullScreen = !workspace.activeWindow.fullScreen;"
    "} else if (workspace.activeClient) {"           # Plasma 5
    " workspace.activeClient.fullScreen = !workspace.activeClient.fullScreen;"
    "}"
)


def toggle_fullscreen() -> None:
    backend = require()
    if backend == "hyprland":
        _run(["hyprctl", "dispatch", "fullscreen", "1"])
    elif backend == "sway":
        _run(["swaymsg", "fullscreen", "toggle"])
    elif backend == "gnome":
        try:
            _gnome_call("ToggleFullscreen")
        except Exception:
            _gnome_eval(
                "let w = global.display.focus_window;"
                "w.fullscreen ? w.unfullscreen() : w.fullscreen(); 'done'"
            )
    elif backend == "x11":
        _run(["wmctrl", "-r", _X11_ACTIVE, "-b", "toggle,fullscreen"])
    elif backend == "kde":
        _kwin_run_script(_KDE_FULLSCREEN_SCRIPT)
    else:
        raise _unsupported(backend)


def minimize_active() -> None:
    backend = require()
    if backend == "x11":
        _x11_minimize()
        return
    # Deliberately unimplemented elsewhere: tiling compositors have no
    # minimize concept, and the richer backends are left honest rather
    # than faked through a generic fallback.
    raise _unsupported(backend)


def focus_address(address: str) -> None:
    backend = require()
    if backend == "hyprland":
        _run(["hyprctl", "dispatch", "focuswindow", address])
    elif backend == "sway":
        _run(["swaymsg", f"[con_id={address}]", "focus"])
    elif backend == "gnome":
        aid = int(address)
        try:
            _gnome_call("Activate", "i", (aid,))
        except Exception:
            _gnome_eval(
                f"let wins = global.get_window_actors().map(a => a.meta_window);"
                f"let w = wins.find(w => {_ID_EXPR} === {aid});"
                f"if (w) w.activate(0); 'done'"
            )
    elif backend == "x11":
        _run(["wmctrl", "-i", "-a", address])
    elif backend == "kde":
        _kde_kdotool("windowactivate", address)
    else:
        raise _unsupported(backend)


def focus_by_token(token: str) -> None:
    """Focus by a fuzzy class/title token (generative-path targets)."""
    backend = require()
    if backend == "hyprland":
        _run(["hyprctl", "dispatch", "focuswindow", token])
        return
    needle = token.lower()
    for w in list_windows():
        hay = f"{w.cls.lower()} {w.title.lower()}"
        if needle in hay:
            focus_address(w.address)
            return
    raise LookupError(f"no window matching {token!r}")


def switch_workspace(n: int) -> None:
    backend = require()
    if backend == "hyprland":
        _run(["hyprctl", "dispatch", "workspace", str(n)])
    elif backend == "sway":
        _run(["swaymsg", "workspace", "number", str(n)])
    elif backend == "gnome":
        try:
            _gnome_call("SetWorkspace", "i", (n,))
        except Exception:
            _gnome_eval(
                f"global.workspace_manager.get_workspace_by_index({n - 1}).activate(0);"
                f" 'done'"
            )
    elif backend == "x11":
        _run(["wmctrl", "-s", str(n - 1)])       # wmctrl desktops are 0-indexed
    elif backend == "kde":
        _kwin_run_script(f"workspace.currentDesktop = {n};")
    else:
        raise _unsupported(backend)
