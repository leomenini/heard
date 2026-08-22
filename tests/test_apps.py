from unittest import mock

from heard.tools.apps import launch_app, _find_window
from heard.tools.types import Ok, Rejected, Failed


WINDOWS = {
    "0x1": "firefox Mozilla Firefox",
    "0x2": "foot terminal",
    "0x3": "org.gnome.Calculator",
}


class TestLaunchApp:
    @mock.patch("heard.tools.apps._running_windows", return_value={})
    @mock.patch("heard.tools.apps.app_exists", return_value=True)
    @mock.patch("heard.tools.apps.subprocess.Popen")
    def test_launch(self, mock_popen, mock_exists, _windows):
        r = launch_app("firefox")
        assert isinstance(r, Ok)
        assert "launched" in r.detail
        mock_popen.assert_called_once()

    @mock.patch("heard.tools.apps._running_windows", return_value={})
    @mock.patch("heard.tools.apps.app_exists", return_value=False)
    def test_app_not_found(self, mock_exists, _windows):
        r = launch_app("nonexistent")
        assert isinstance(r, Rejected)
        assert r.kind == "declined"

    @mock.patch("heard.tools.apps._running_windows", return_value={})
    @mock.patch("heard.tools.apps.app_exists", return_value=True)
    @mock.patch("heard.tools.apps.subprocess.Popen", side_effect=FileNotFoundError("uwsm"))
    def test_missing_uwsm(self, mock_popen, mock_exists, _windows):
        r = launch_app("firefox")
        assert isinstance(r, Rejected)
        assert r.kind == "declined"

    @mock.patch("heard.tools.apps._running_windows", return_value=WINDOWS)
    @mock.patch("heard.tools.apps.app_exists", return_value=True)
    @mock.patch("heard.tools.apps.subprocess.run")
    def test_focuses_running_instance(self, mock_run, mock_exists, _windows):
        r = launch_app("firefox")
        assert isinstance(r, Ok)
        assert "focused" in r.detail
        args = mock_run.call_args[0][0]
        assert args[:3] == ["hyprctl", "dispatch", "focuswindow"]
        assert args[3] == "0x1"
        # class-name match despite different casing/format
        r2 = launch_app("gnome-calculator")
        assert isinstance(r2, Ok)
        assert "focused" in r2.detail

    @mock.patch("heard.tools.apps._running_windows", return_value=WINDOWS)
    @mock.patch("heard.tools.apps.app_exists", return_value=True)
    @mock.patch("heard.tools.apps.subprocess.Popen")
    def test_launches_when_not_running(self, mock_popen, mock_exists, _windows):
        r = launch_app("spotify")
        assert isinstance(r, Ok)
        assert "launched" in r.detail

    @mock.patch("heard.tools.apps._running_windows",
                side_effect=RuntimeError("hyprctl exploded"))
    @mock.patch("heard.tools.apps.app_exists", return_value=True)
    @mock.patch("heard.tools.apps.subprocess.Popen")
    def test_focus_failure_still_launches(self, mock_popen, mock_exists, _windows):
        r = launch_app("firefox")
        assert isinstance(r, Ok)
        assert "launched" in r.detail


class TestFindWindow:
    def test_matches_class_token(self):
        assert _find_window(["calculator"], WINDOWS) == "0x3"

    def test_matches_title_words(self):
        assert _find_window(["mozilla"], WINDOWS) == "0x1"

    def test_no_match_below_cutoff(self):
        assert _find_window(["zzzz"], WINDOWS) is None

    def test_empty_windows(self):
        assert _find_window(["firefox"], {}) is None
