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

        # direct: switch_workspace routes to _kwin_run_script on kde
        with mock.patch.object(wm, "require", lambda: "kde"), \
             mock.patch.object(wm, "_kwin_run_script", fake_run):
            wm.switch_workspace(5)
        assert "workspace.currentDesktop = 5" in captured["src"]

    def test_toggle_fullscreen_script_content(self):
        captured = {}

        def fake_run(source):
            captured["src"] = source

        with mock.patch.object(wm, "require", lambda: "kde"), \
             mock.patch.object(wm, "_kwin_run_script", fake_run):
            wm.toggle_fullscreen()
        assert "fullScreen = !workspace.activeWindow.fullScreen" in captured["src"]
        assert "activeClient" in captured["src"]  # Plasma 5 branch


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


class TestGnome:
    @pytest.fixture(autouse=True)
    def _reset_extension_latch(self):
        wm._gnome_extension_dead = False
        yield
        wm._gnome_extension_dead = False

    def test_detect_from_desktop(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "c.toml"))
        from heard import config as cfg
        cfg._cached_load.cache_clear()
        try:
            monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
            monkeypatch.delenv("SWAYSOCK", raising=False)
            monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
            assert wm.detect() == "gnome"
        finally:
            cfg._cached_load.cache_clear()

    # --- extension tier ---------------------------------------------------

    def test_list_via_extension(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "gnome")
        payload = json.dumps([{"id": 42, "cls": "firefox", "title": "Mozilla Firefox"}])
        with mock.patch.object(wm, "_gnome_call", return_value=payload) as call, \
             mock.patch.object(wm, "_gnome_eval") as ev:
            windows = wm.list_windows()
        assert windows == [Window("42", "firefox", "Mozilla Firefox")]
        assert call.call_args[0][0] == "ListWindows"
        ev.assert_not_called()

    def test_close_active_routes_extension(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "gnome")
        with mock.patch.object(wm, "_gnome_call") as call, \
             mock.patch.object(wm, "_gnome_eval") as ev:
            wm.close_active()
        assert call.call_args[0] == ("Close", "s", ("active",))
        ev.assert_not_called()

    def test_toggle_fullscreen_routes_extension(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "gnome")
        with mock.patch.object(wm, "_gnome_call") as call, \
             mock.patch.object(wm, "_gnome_eval") as ev:
            wm.toggle_fullscreen()
        assert call.call_args[0] == ("ToggleFullscreen",)
        ev.assert_not_called()

    def test_focus_address_routes_extension(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "gnome")
        with mock.patch.object(wm, "_gnome_call") as call, \
             mock.patch.object(wm, "_gnome_eval") as ev:
            wm.focus_address("42")
        assert call.call_args[0] == ("Activate", "i", (42,))
        ev.assert_not_called()

    def test_switch_workspace_routes_extension(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "gnome")
        with mock.patch.object(wm, "_gnome_call") as call, \
             mock.patch.object(wm, "_gnome_eval") as ev:
            wm.switch_workspace(3)
        assert call.call_args[0] == ("SetWorkspace", "i", (3,))
        ev.assert_not_called()

    # --- eval fallback tier -------------------------------------------------

    def test_list_falls_back_to_eval(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "gnome")
        payload = json.dumps([{"id": 7, "cls": "foot", "title": "term"}])
        with mock.patch.object(wm, "_gnome_call",
                               side_effect=RuntimeError("absent")), \
             mock.patch.object(wm, "_gnome_eval", return_value=payload) as ev:
            windows = wm.list_windows()
        assert windows == [Window("7", "foot", "term")]
        assert "get_window_actors" in ev.call_args[0][0]

    def test_close_active_falls_back_to_eval(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "gnome")
        with mock.patch.object(wm, "_gnome_call",
                               side_effect=RuntimeError("absent")), \
             mock.patch.object(wm, "_gnome_eval") as ev:
            wm.close_active()
        assert "delete(0)" in ev.call_args[0][0]

    def test_focus_address_falls_back_to_eval(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "gnome")
        with mock.patch.object(wm, "_gnome_call",
                               side_effect=RuntimeError("absent")), \
             mock.patch.object(wm, "_gnome_eval") as ev:
            wm.focus_address("42")
        js = ev.call_args[0][0]
        assert "=== 42" in js

    def test_switch_workspace_falls_back_to_eval(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "gnome")
        with mock.patch.object(wm, "_gnome_call",
                               side_effect=RuntimeError("absent")), \
             mock.patch.object(wm, "_gnome_eval") as ev:
            wm.switch_workspace(3)
        assert "get_workspace_by_index(2)" in ev.call_args[0][0]

    # --- shared parsing + latch ----------------------------------------------

    def test_windows_from_json_empty_string(self):
        assert wm._windows_from_json("") == []
        assert wm._windows_from_json("  ") == []

    def test_extension_failure_latches(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "gnome")
        err = mock.Mock()
        err.message_type = 4                     # D-Bus error reply
        conn = mock.Mock()
        conn.send_message.return_value = err

        with mock.patch.object(wm, "_dbus_session_connection",
                               return_value=conn), \
             mock.patch.object(wm, "_gnome_eval", return_value="[]"):
            wm.list_windows()          # first attempt fails -> latch
            wm.list_windows()          # second skips the dbus path entirely
        assert conn.send_message.call_count == 1


class TestWindowLabel:
    def test_label_combines_class_and_title(self):
        assert Window("1", "Foot", "Terminal").label() == "foot terminal"
