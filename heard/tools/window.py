import subprocess

from .helpers import wm
from .types import Failed, Ok, Rejected, Result


def window_action(action: str, target: str | None = None) -> Result:
    if action not in ("close", "focus", "fullscreen", "minimize"):
        return Rejected("window_action", f"bad action {action!r}", "invalid_value")
    if action == "focus" and target is None:
        return Rejected("window_action", "focus needs a target", "bad_args")

    try:
        if action == "close":
            wm.close_active()
        elif action == "fullscreen":
            try:
                wm.toggle_fullscreen()
            except NotImplementedError as e:
                return Failed("window_action", str(e))
        elif action == "minimize":
            wm.minimize_active()
        else:
            assert target is not None  # guarded at function entry
            wm.focus_by_token(target)
        return Ok("window_action", f"window {action}")
    except FileNotFoundError as e:
        name = getattr(e, "filename", None)
        return Failed("window_action", f"{name or 'required binary'} not found")
    except subprocess.CalledProcessError as e:
        binary = e.cmd[0] if getattr(e, "cmd", None) else "backend command"
        return Failed("window_action", f"{binary} exited {e.returncode}")
    except (RuntimeError, LookupError) as e:
        return Failed("window_action", str(e))
