"""Window-manager abstraction: Hyprland, Sway, KDE, and GNOME.

Detection order: config `wm_backend` override, then environment
(HYPRLAND_INSTANCE_SIGNATURE / SWAYSOCK / KDE / GNOME markers). Every command
raises on failure -- handlers translate that into Failed/Rejected results.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

from jeepney import DBusAddress, new_method_call

BACKENDS = ("hyprland", "sway", "kde", "gnome")
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


def _gnome_eval(js: str) -> str:
    """Run JS inside GNOME Shell via D-Bus Eval; return its string result."""
    from jeepney.io.blocking import open_dbus_connection

    try:
        from .. import media

        conn = media._connection()
    except Exception:
        conn = open_dbus_connection()
    reply = conn.send_message(new_method_call(_GNOME_SHELL, "Eval", "s", (js,)),
                              timeout=_DBUS_TIMEOUT_S)
    if reply is None or reply.message_type == 4:  # error
        raise RuntimeError("gnome shell eval failed")
    success, result = reply.body
    if not success:
        raise RuntimeError(f"gnome shell eval error: {result}")
    return result


def _gnome_list() -> list[Window]:
    js = (
        "JSON.stringify(global.get_window_actors().map(a => ({"
        "id: a.meta_window.get_id(),"
        "cls: a.meta_window.get_wm_class() || '',"
        "title: a.meta_window.get_title() || ''"
        "})))"
    )
    out = _gnome_eval(js)
    windows = []
    for item in json.loads(out) if out.strip() else []:
        windows.append(Window(str(item["id"]), str(item.get("cls", "")),
                              str(item.get("title", ""))))
    return windows


# --- KDE -------------------------------------------------------------------

_KWIN_SCRIPTING = DBusAddress("/KWin", bus_name="org.kde.KWin",
                              interface="org.kde.KWin.Scripting")


def _dbus_call(address: DBusAddress, method: str, signature=None, body=None):
    from jeepney.io.blocking import open_dbus_connection

    try:
        # reuse the persistent session-bus connection cached by the MPRIS module
        from .. import media

        conn = media._connection()
    except Exception:
        conn = open_dbus_connection()
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


# --- Public API ------------------------------------------------------------

def list_windows() -> list[Window]:
    backend = require()
    if backend == "hyprland":
        return _hyprland_list()
    if backend == "sway":
        return _sway_list()
    if backend == "gnome":
        return _gnome_list()
    return _kde_list()


def close_active() -> None:
    backend = require()
    if backend == "hyprland":
        _run(["hyprctl", "dispatch", "closewindow", "active"])
    elif backend == "sway":
        _run(["swaymsg", "kill"])
    elif backend == "gnome":
        _gnome_eval("global.display.focus_window.delete(0); 'done'")
    else:
        active = _kde_kdotool("getactivewindow")
        _kde_kdotool("windowclose", active)


def toggle_fullscreen() -> None:
    backend = require()
    if backend == "hyprland":
        _run(["hyprctl", "dispatch", "fullscreen", "1"])
    elif backend == "sway":
        _run(["swaymsg", "fullscreen", "toggle"])
    elif backend == "gnome":
        _gnome_eval(
            "let w = global.display.focus_window;"
            "w.fullscreen ? w.unfullscreen() : w.fullscreen(); 'done'"
        )
    else:
        raise NotImplementedError(
            "fullscreen toggling is not supported on KDE yet")


def focus_address(address: str) -> None:
    backend = require()
    if backend == "hyprland":
        _run(["hyprctl", "dispatch", "focuswindow", address])
    elif backend == "sway":
        _run(["swaymsg", f"[con_id={address}]", "focus"])
    elif backend == "gnome":
        _gnome_eval(
            f"let wins = global.get_window_actors().map(a => a.meta_window);"
            f"let w = wins.find(w => w.get_id() === {address});"
            f"if (w) w.activate(0); 'done'"
        )
    else:
        _kde_kdotool("windowactivate", address)


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
        _gnome_eval(
            f"global.workspace_manager.get_workspace_by_index({n - 1}).activate(0);"
            f" 'done'"
        )
    else:
        _kwin_run_script(f"workspace.currentDesktop = {n};")
