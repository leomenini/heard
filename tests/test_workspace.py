from unittest import mock

import pytest

from heard.tools.types import Failed, Ok, Rejected
from heard.tools.workspace import workspace_switch


@pytest.fixture
def hyprland(monkeypatch):
    monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "hyprland")
    return "heard.tools.helpers.wm._run"


class TestWorkspaceSwitch:
    def test_valid(self, hyprland):
        with mock.patch(hyprland) as run:
            r = workspace_switch("3")
        assert isinstance(r, Ok)
        assert ["hyprctl", "dispatch", "workspace", "3"] == run.call_args[0][0]

    @pytest.mark.parametrize("n", ("1", "10"))
    def test_range_bounds(self, hyprland, n):
        with mock.patch(hyprland):
            assert isinstance(workspace_switch(n), Ok)

    @pytest.mark.parametrize("bad", ("99", "0"))
    def test_out_of_range(self, bad):
        r = workspace_switch(bad)
        assert isinstance(r, Rejected)
        assert r.kind == "invalid_value"

    def test_not_a_number(self):
        r = workspace_switch("abc")
        assert isinstance(r, Rejected)
        assert r.kind == "invalid_value"

    def test_missing_binary(self, hyprland):
        with mock.patch(hyprland, side_effect=FileNotFoundError("hyprctl")):
            r = workspace_switch("3")
        assert isinstance(r, Failed)

    def test_sway_uses_number_keyword(self, monkeypatch):
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "sway")
        with mock.patch("heard.tools.helpers.wm._run") as run:
            r = workspace_switch("3")
        assert isinstance(r, Ok)
        assert run.call_args[0][0] == ["swaymsg", "workspace", "number", "3"]

    def test_kde_runs_kwin_script(self, monkeypatch):
        wm = "heard.tools.helpers.wm"
        monkeypatch.setattr(f"{wm}.require", lambda: "kde")
        with mock.patch(f"{wm}._kwin_run_script") as script:
            r = workspace_switch("3")
        assert isinstance(r, Ok)
        assert "workspace.currentDesktop = 3" in script.call_args[0][0]
