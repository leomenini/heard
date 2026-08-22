from unittest import mock

import pytest

from heard.tools.types import Failed, Ok, Rejected
from heard.tools.window import window_action


@pytest.fixture
def hyprland(monkeypatch):
    monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "hyprland")
    return "heard.tools.helpers.wm._run"


class TestWindowAction:
    def test_close(self, hyprland):
        with mock.patch(hyprland) as run:
            r = window_action("close")
        assert isinstance(r, Ok)
        assert ["hyprctl", "dispatch", "closewindow", "active"] == run.call_args[0][0]

    def test_fullscreen(self, hyprland):
        with mock.patch(hyprland):
            r = window_action("fullscreen")
        assert isinstance(r, Ok)

    def test_focus(self, hyprland):
        with mock.patch(hyprland) as run:
            r = window_action("focus", target="firefox")
        assert isinstance(r, Ok)
        assert run.call_args[0][0][-1] == "firefox"

    def test_focus_no_target(self):
        assert isinstance(window_action("focus"), Rejected)

    def test_bad_action(self):
        r = window_action("shutdown")
        assert isinstance(r, Rejected)
        assert r.kind == "invalid_value"

    def test_missing_binary(self, hyprland):
        with mock.patch(hyprland, side_effect=FileNotFoundError("hyprctl")):
            r = window_action("close")
        assert isinstance(r, Failed)
        assert "not found" in r.reason

    def test_nonzero_exit(self, hyprland):
        import subprocess

        err = subprocess.CalledProcessError(1, ["hyprctl", "dispatch"])
        with mock.patch(hyprland, side_effect=err):
            r = window_action("close")
        assert isinstance(r, Failed)


class TestBackendDispatch:
    def test_sway_close(self, monkeypatch):
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "sway")
        with mock.patch("heard.tools.helpers.wm._run") as run:
            r = window_action("close")
        assert isinstance(r, Ok)
        assert run.call_args[0][0] == ["swaymsg", "kill"]

    def test_kde_close(self, monkeypatch):
        wm = "heard.tools.helpers.wm"
        monkeypatch.setattr(f"{wm}.require", lambda: "kde")
        with mock.patch(f"{wm}._kde_kdotool", return_value="944") as kdo:
            r = window_action("close")
        assert isinstance(r, Ok)
        assert ("windowclose", "944") == kdo.call_args[0]

    def test_kde_fullscreen_unsupported(self, monkeypatch):
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "kde")
        r = window_action("fullscreen")
        assert isinstance(r, Failed)
        assert "KDE" in r.reason

    def test_no_wm_detected_fails_with_message(self, monkeypatch):
        monkeypatch.setattr("heard.tools.helpers.wm.detect", lambda: None)
        r = window_action("close")
        assert isinstance(r, Failed)
        assert "window manager" in r.reason
