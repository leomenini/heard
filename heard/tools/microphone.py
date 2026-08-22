"""Microphone mute/unmute control."""

import subprocess

from .types import Failed, Ok, Rejected, Result

DEFAULT_SOURCE = "@DEFAULT_AUDIO_SOURCE@"


def _set_source_mute(mute: bool) -> Result:
    value = "1" if mute else "0"
    try:
        subprocess.run(
            ["wpctl", "set-mute", DEFAULT_SOURCE, value],
            check=True,
        )
    except FileNotFoundError:
        return Failed("microphone_control", "wpctl not found")
    except subprocess.CalledProcessError as e:
        return Failed("microphone_control", f"wpctl failed: {e}")
    return Ok("microphone_control", "muted" if mute else "unmuted")


def microphone_control(action: str) -> Result:
    """Mute or unmute the default microphone source."""
    if action == "mute":
        return _set_source_mute(True)
    if action == "unmute":
        return _set_source_mute(False)
    return Rejected(
        "microphone_control",
        f"invalid action {action!r}; expected mute or unmute",
        "invalid_value",
    )
