import subprocess

from rapidfuzz import fuzz

from .helpers import wm
from .helpers.apps import app_exists, installed_apps, resolve_app
from .types import Ok, Rejected, Result


def _find_window(queries: list[str], windows: list[wm.Window]) -> str | None:
    """Fuzzy-match spoken phrase / binary against open windows."""
    best_addr, best_score = None, 0.0
    for w in windows:
        for q in queries:
            score = max(fuzz.token_set_ratio(q.lower(), w.label()),
                        fuzz.partial_ratio(q.lower(), w.label()))
            if score > best_score:
                best_addr, best_score = w.address, score
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
        windows = wm.list_windows()
        addr = _find_window(queries, windows)
        if addr is not None:
            wm.focus_address(addr)
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
