import subprocess

from .types import Failed, Ok, Rejected, Result

STEP = "5%"


def _set_volume(action: str, amount: str | None, sink: str) -> Result:
    if action == "up":
        cmd = ["wpctl", "set-volume", sink, f"{STEP}+"]
    elif action == "down":
        cmd = ["wpctl", "set-volume", sink, f"{STEP}-"]
    elif action == "mute":
        cmd = ["wpctl", "set-mute", sink, "1"]
    elif action == "unmute":
        cmd = ["wpctl", "set-mute", sink, "0"]
    elif action == "set":
        try:
            pct = int(str(amount))
        except (TypeError, ValueError):
            return Rejected("volume_control",
                            f"invalid amount {amount!r}", "invalid_value")
        cmd = ["wpctl", "set-volume", sink, f"{pct}%"]
    else:
        return Rejected("volume_control", f"bad action {action!r}", "invalid_value")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        return Failed("volume_control", "wpctl not found")
    return Ok("volume_control", f"volume {action}")


def volume_control(action: str, amount: str | None = None) -> Result:
    # amount is only meaningful for 'set'; junk slots from the model are dropped
    return _set_volume(action, amount if action == "set" else None,
                       "@DEFAULT_AUDIO_SINK@")
