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

    def test_minimize(self, monkeypatch):
        """Minimize is x11-only; the happy path goes through _x11_minimize."""
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "x11")
        with mock.patch("heard.tools.helpers.wm._x11_minimize") as mini:
            r = window_action("minimize")
        assert isinstance(r, Ok)
        assert r.detail == "window minimize"
        mini.assert_called_once()

    def test_minimize_unsupported_backend_fails(self, monkeypatch):
        """hyprland has no minimize: the handler reports Failed, not a crash."""
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "hyprland")
        monkeypatch.setattr(
            "heard.tools.helpers.wm._unsupported",
            lambda b: RuntimeError(f"unsupported wm backend {b!r}"),
        )
        r = window_action("minimize")
        assert isinstance(r, Failed)
        assert "unsupported" in r.reason

    def test_minimize_missing_dependency_fails(self, monkeypatch, tmp_path):
        """python-xlib absent on x11: RuntimeError -> Failed, never raises."""
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "x11")
        monkeypatch.setattr(
            "heard.tools.helpers.wm._x11_minimize",
            mock.Mock(side_effect=RuntimeError("python-xlib is required")),
        )
        r = window_action("minimize")
        assert isinstance(r, Failed)
        assert "python-xlib" in r.reason


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

    def test_kde_fullscreen_via_kwin_script(self, monkeypatch):
        wm = "heard.tools.helpers.wm"
        monkeypatch.setattr(f"{wm}.require", lambda: "kde")
        with mock.patch(f"{wm}._kwin_run_script") as script:
            r = window_action("fullscreen")
        assert isinstance(r, Ok)
        assert "fullScreen" in script.call_args[0][0]

    def test_no_wm_detected_fails_with_message(self, monkeypatch):
        monkeypatch.setattr("heard.tools.helpers.wm.detect", lambda: None)
        r = window_action("close")
        assert isinstance(r, Failed)
        assert "window manager" in r.reason
