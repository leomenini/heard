import subprocess
from functools import lru_cache

from jeepney import DBusAddress, new_method_call
from jeepney.io.blocking import open_dbus_connection

from .types import Failed, Ok, Rejected, Result

METHODS = {"play": "Play", "pause": "Pause", "next": "Next", "previous": "Previous"}
MPRIS_PREFIX = "org.mpris.MediaPlayer2."
DBUS_TIMEOUT_S = 1.0

_DBUS = DBusAddress(
    "/org/freedesktop/DBus",
    bus_name="org.freedesktop.DBus",
    interface="org.freedesktop.DBus",
)


@lru_cache(maxsize=1)
def _connection():
    """Session bus connection, opened once and reused across commands."""
    return open_dbus_connection()


def _players() -> list[DBusAddress]:
    conn = _connection()
    reply = conn.send_message(new_method_call(_DBUS, "ListNames"), timeout=DBUS_TIMEOUT_S)
    names = reply.body[0] if reply else []
    return [
        DBusAddress("/org/mpris/MediaPlayer2", bus_name=name,
                    interface=f"{MPRIS_PREFIX}Player")
        for name in names
        if isinstance(name, str) and name.startswith(MPRIS_PREFIX)
    ]


def _mpris_send(action: str) -> bool:
    """Send the transport command to every MPRIS player over one connection.

    Returns False (and clears the cached bus) whenever D-Bus is unavailable,
    letting the caller fall back to playerctl.
    """
    try:
        players = _players()
        if not players:
            return False
        conn = _connection()
        for player in players:
            conn.send_message(
                new_method_call(player, METHODS[action]), timeout=DBUS_TIMEOUT_S
            )
        return True
    except Exception:
        _connection.cache_clear()
        return False


def _playerctl(action: str) -> Result:
    cmd = ["playerctl", action]
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return Failed("media_control", "playerctl not found")
    except subprocess.CalledProcessError as e:
        return Failed("media_control", f"no media player running (rc={e.returncode})")
    return Ok("media_control", f"media {action}")


def media_control(action: str) -> Result:
    if action not in METHODS:
        return Rejected("media_control", f"bad action {action!r}", "invalid_value")
    if _mpris_send(action):
        return Ok("media_control", f"media {action} [mpris]")
    return _playerctl(action)
