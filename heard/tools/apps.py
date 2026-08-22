import json
import re
import subprocess

from rapidfuzz import fuzz, process

from .helpers.apps import app_exists, installed_apps, resolve_app
from .types import Ok, Rejected, Result


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _running_windows() -> dict[str, str]:
    """address -> searchable label for every open Hyprland window."""
    try:
        out = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True,
                             text=True, timeout=2)
        if out.returncode != 0:
            return {}
        clients = json.loads(out.stdout)
    except Exception:
        return []
    windows: dict[str, str] = {}
    for w in clients if isinstance(clients, list) else []:
        addr = w.get("address")
        label = _norm(" ".join(filter(None, (
            w.get("class"), w.get("initialClass"),
            w.get("title"), w.get("initialTitle"),
        ))))
        if addr and label:
            windows[addr] = label
    return windows


def _find_window(queries: list[str], windows: dict[str, str]) -> str | None:
    """Fuzzy-match spoken phrase / binary against open windows."""
    if not windows:
        return None
    best_addr, best_score = None, 0.0
    for addr, label in windows.items():
        for q in queries:
            score = max(fuzz.token_set_ratio(_norm(q), label),
                        fuzz.partial_ratio(_norm(q), label))
            if score > best_score:
                best_addr, best_score = addr, score
    return best_addr if best_score >= 80 else None


def _display_name(binary: str) -> str | None:
    for name, b in installed_apps().items():
        if b == binary and len(name) > len(binary):
            return name
    return None


def launch_app(app: str) -> Result:
    binary = app if app_exists(app) else resolve_app(app)
    if binary is None:
        return Rejected("launch_app", f"no such app {app!r}", "declined")

    # already running? focus it instead of spawning a second instance
    queries = [q for q in (app, binary, _display_name(binary)) if q]
    try:
        addr = _find_window(queries, _running_windows())
        if addr is not None:
            subprocess.run(["hyprctl", "dispatch", "focuswindow", addr],
                           check=True, timeout=2)
            return Ok("launch_app", f"focused running {binary}")
    except FileNotFoundError:
        pass
    except Exception:
        pass  # focus failed; a launch is still better than nothing

    try:
        subprocess.Popen(
            ["uwsm", "app", "--", binary],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return Ok("launch_app", f"launched {binary}")
    except FileNotFoundError:
        return Rejected("launch_app", "uwsm not found", "declined")
