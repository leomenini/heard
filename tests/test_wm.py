import json
from unittest import mock

import pytest

from heard.tools.helpers import wm
from heard.tools.helpers.wm import Window


class TestDetect:
    def test_env_hyprland(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "c.toml"))
        from heard import config as cfg
        cfg._cached_load.cache_clear()
        monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
        monkeypatch.delenv("SWAYSOCK", raising=False)
        try:
            monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "sig")
            assert wm.detect() == "hyprland"
            monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE")
            monkeypatch.setenv("SWAYSOCK", "/run/sway.sock")
            assert wm.detect() == "sway"
            monkeypatch.delenv("SWAYSOCK")
            assert wm.detect() is None
        finally:
            cfg._cached_load.cache_clear()

    def test_config_override_wins(self, monkeypatch, tmp_path):
        c = tmp_path / "c.toml"
        c.write_text('wm_backend = "sway"\n')
        monkeypatch.setenv("HEARD_CONFIG", str(c))
        from heard import config as cfg
        cfg._cached_load.cache_clear()
        try:
            monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "sig")
            assert wm.detect() == "sway"
        finally:
            cfg._cached_load.cache_clear()


class TestSwayTree:
    TREE = {
        "id": 1, "type": "root", "name": "root",
        "nodes": [{
            "id": 2, "type": "output", "name": "eDP-1",
            "nodes": [{
                "id": 3, "type": "workspace",
                "nodes": [
                    {"id": 7, "type": "con", "name": "Firefox",
                     "app_id": "firefox"},
                    {"id": 8, "type": "floating_con", "name": "Calc",
                     "app_id": None,
                     "window_properties": {"class": "Gnome-calculator"}},
                ],
                "floating_nodes": [],
            }],
        }],
    }

    def test_walk_collects_leaf_windows(self):
        out: list[Window] = []
        wm._sway_walk(self.TREE, out)
        pairs = sorted((w.address, w.cls) for w in out)
        assert pairs == [("7", "firefox"), ("8", "Gnome-calculator")]

    def test_sway_list_parses_output(self, monkeypatch):
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "sway")
        raw = json.dumps(self.TREE)
        with mock.patch("heard.tools.helpers.wm._run") as run:
            run.return_value = mock.Mock(stdout=raw)
            windows = wm.list_windows()
        assert len(windows) == 2


class TestHyprlandList:
    def test_parses_clients(self, monkeypatch):
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "hyprland")
        clients = [{"address": "0x1", "initialClass": "foot", "title": "term"}]
        with mock.patch("heard.tools.helpers.wm._run") as run:
            run.return_value = mock.Mock(stdout=json.dumps(clients))
            windows = wm.list_windows()
        assert windows == [Window("0x1", "foot", "term")]


class TestKde:
    def test_list_via_kdotool(self, monkeypatch):
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "kde")

        def fake_kdotool(*args):
            if args[:1] == ("search",):
                return "944\n945"
            names = {"944": "Firefox", "945": "Term"}
            classes = {"944": "firefox", "945": "foot"}
            if args[0] == "getwindowname":
                return names.get(args[1], "")
            return classes.get(args[1], "")

        with mock.patch("heard.tools.helpers.wm._kde_kdotool", side_effect=fake_kdotool):
            windows = wm.list_windows()
        assert Window("944", "firefox", "Firefox") in windows

    def test_workspace_script_content(self):
        captured = {}

        def fake_run(source):
            captured["src"] = source

        with mock.patch("heard.tools.helpers.wm._kwin_run_script", side_effect=fake_run):
            wm.switch_workspace_kde = None  # not used; direct call below

        # direct: switch_workspace routes to _kwin_run_script on kde
        with mock.patch.object(wm, "require", lambda: "kde"), \
             mock.patch.object(wm, "_kwin_run_script", fake_run):
            wm.switch_workspace(5)
        assert "workspace.currentDesktop = 5" in captured["src"]


class TestFocusByToken:
    def test_matches_window_label(self, monkeypatch):
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "sway")
        wins = [Window("11", "firefox", "Mozilla Firefox"),
                Window("12", "foot", "terminal")]
        sent = []

        def record(cmd):
            sent.append(cmd)

        with mock.patch.object(wm, "list_windows", return_value=wins), \
             mock.patch.object(wm, "_run", side_effect=record):
            wm.focus_by_token("mozilla")
        assert sent == [["swaymsg", "[con_id=11]", "focus"]]

    def test_no_match_raises(self, monkeypatch):
        monkeypatch.setattr("heard.tools.helpers.wm.require", lambda: "sway")
        with mock.patch.object(wm, "list_windows", return_value=[]), \
             pytest.raises(LookupError):
            wm.focus_by_token("nothing-here")


class TestWindowLabel:
    def test_label_combines_class_and_title(self):
        assert Window("1", "Foot", "Terminal").label() == "foot terminal"
