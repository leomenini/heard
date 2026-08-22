"""Shared fixtures and mocks for all tests.

Heavy modules (evdev, sounddevice, faster_whisper, needle) are mocked
at the module level so importing heard modules doesn't trigger hardware
access or expensive model loads.
"""

import sys
from unittest import mock

import pytest

# Mock heavy dependencies before any heard import touches them
for mod_name in ("evdev", "sounddevice", "faster_whisper", "needle"):
    sys.modules[mod_name] = mock.MagicMock()


@pytest.fixture(autouse=True)
def _never_touch_live_compositor(monkeypatch):
    """Block every path that could reach the real window manager.

    Tests that need WM behavior must mock explicitly (wm._run,
    wm._kde_kdotool, wm._dbus_call); anything unmocked explodes loudly
    instead of closing the developer's windows.
    """
    from heard.tools.helpers import wm

    def _blocked(name):
        def boom(*a, **k):
            raise AssertionError(
                f"test tried to hit the live session via {name}; "
                "mock it explicitly"
            )
        return boom

    monkeypatch.setattr(wm, "_run", _blocked("wm._run"))
    monkeypatch.setattr(wm, "_kde_kdotool", _blocked("wm._kde_kdotool"))
    monkeypatch.setattr(wm, "_dbus_call", _blocked("wm._dbus_call"))

# Give ecodes realistic behavior: known keys resolve to ints, unknown raise
ecodes = mock.Mock(spec_set=["EV_KEY", "KEY_LEFTSHIFT", "KEY_LEFTCONTROL", "KEY_ESC"])
ecodes.EV_KEY = 1
ecodes.KEY_LEFTSHIFT = 42
ecodes.KEY_LEFTCONTROL = 29
ecodes.KEY_ESC = 1
sys.modules["evdev"].ecodes = ecodes

# Patch list_devices to return empty (no input devices in CI)
sys.modules["evdev"].list_devices.return_value = []
