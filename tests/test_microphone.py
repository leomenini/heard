"""Microphone mute/unmute tool."""

from unittest import mock

from heard.tools.microphone import microphone_control
from heard.tools.types import Ok, Rejected


@mock.patch("heard.tools.microphone.subprocess.run")
def test_mute_runs_wpctl(mock_run):
    r = microphone_control("mute")
    assert isinstance(r, Ok)
    assert r.tool == "microphone_control"
    cmd = mock_run.call_args[0][0]
    assert cmd[:3] == ["wpctl", "set-mute", "@DEFAULT_AUDIO_SOURCE@"]
    assert cmd[3] == "1"


@mock.patch("heard.tools.microphone.subprocess.run")
def test_unmute_runs_wpctl(mock_run):
    r = microphone_control("unmute")
    assert isinstance(r, Ok)
    cmd = mock_run.call_args[0][0]
    assert cmd[3] == "0"


def test_invalid_action_rejected():
    r = microphone_control("up")
    assert isinstance(r, Rejected)
    assert r.kind == "invalid_value"


@mock.patch("heard.tools.microphone.subprocess.run",
            side_effect=FileNotFoundError("wpctl"))
def test_missing_wpctl_fails_gracefully(mock_run):
    r = microphone_control("mute")
    assert r.tool == "microphone_control"
    assert "wpctl not found" in r.reason
