"""Screen recording tool.

Uses ``wf-recorder`` on Wayland (best for wlroots-based compositors) and falls
back to ``ffmpeg -f x11grab`` on X11. Recording state is tracked via a pidfile
so ``screenrecord stop`` can finalise the file.
"""

import datetime
import os
import signal
import subprocess
import time
from pathlib import Path

from .types import Failed, Ok, Rejected, Result

PIDFILE = Path.home() / ".local" / "state" / "heard" / "screenrecord.pid"


def _config_value(key: str) -> str | bool:
    from .. import config

    return config.cached(key)


def _recording_state() -> dict | None:
    if not PIDFILE.is_file():
        return None
    try:
        import json
        with open(PIDFILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _save_state(state: dict) -> None:
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(PIDFILE, "w", encoding="utf-8") as f:
        json.dump(state, f)


def _clear_state() -> None:
    try:
        PIDFILE.unlink()
    except FileNotFoundError:
        pass


def _resolve_output(screen: str | None) -> str | None:
    """Map a spoken screen label to a display output name."""
    mapping = str(_config_value("screenrecord_outputs") or "")
    if mapping and screen:
        for pair in mapping.split(","):
            if "=" not in pair:
                continue
            name, output = pair.split("=", 1)
            if name.strip().lower() == screen.strip().lower():
                return output.strip()
    # No mapping matched: treat the spoken label itself as the output name.
    if screen:
        return screen.strip()
    default = str(_config_value("screenrecord_output") or "")
    return default if default else None


def _folder() -> Path:
    folder = str(_config_value("screenrecord_folder") or "~/Videos")
    return Path(folder).expanduser()


def _wf_recorder_cmd(output: str | None, path: Path) -> list[str]:
    cmd = ["wf-recorder", "-f", str(path)]
    if output:
        cmd += ["-o", output]
    return cmd


def _ffmpeg_cmd(path: Path) -> list[str]:
    # X11 fallback; works on Xorg and captures the primary XWayland display.
    # The grab target follows $DISPLAY rather than assuming :0.0.
    return [
        "ffmpeg",
        "-y",
        "-f", "x11grab",
        "-i", os.environ.get("DISPLAY", ":0"),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        str(path),
    ]


def _start_process(cmd: list[str]) -> subprocess.Popen:
    # Detach from the daemon so SIGINTs to us don't stop the recorder.
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _start_recording(output: str | None) -> Result:
    state = _recording_state()
    if state is not None and _process_alive(state.get("pid", -1)):
        return Failed("screen_record", "already recording")

    folder = _folder()
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"screenrecord_{timestamp}.mp4"

    # Prefer wf-recorder on Wayland; fall back to ffmpeg on X11.
    try:
        proc = _start_process(_wf_recorder_cmd(output, path))
    except FileNotFoundError:
        try:
            proc = _start_process(_ffmpeg_cmd(path))
        except FileNotFoundError:
            return Failed("screen_record", "wf-recorder or ffmpeg not found")

    # Give the process a moment to fail on bad arguments; if it died early,
    # report the failure instead of leaving a stale pidfile.
    time.sleep(0.3)
    if proc.poll() is not None:
        return Failed("screen_record", "recorder exited immediately")

    _save_state({"pid": proc.pid, "path": str(path)})
    return Ok("screen_record", f"recording to {path}")


def _stop_recording() -> Result:
    state = _recording_state()
    if state is None:
        return Failed("screen_record", "not recording")

    pid = state.get("pid")
    path = state.get("path", "unknown")
    if pid is not None and _process_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        # Wait briefly for a clean shutdown, then force-kill if necessary.
        deadline = time.time() + 5
        while time.time() < deadline and _process_alive(pid):
            time.sleep(0.1)
        if _process_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    _clear_state()
    return Ok("screen_record", f"saved {path}")


def screen_record(action: str, screen: str | None = None) -> Result:
    """Start or stop a screen recording."""
    if action == "start":
        return _start_recording(_resolve_output(screen))
    if action == "stop":
        return _stop_recording()
    return Rejected(
        "screen_record",
        f"invalid action {action!r}; expected start or stop",
        "invalid_value",
    )
