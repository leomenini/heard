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
        # the generic X11 fallback would otherwise answer for the None case
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
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


class TestX11:
    """Generic EWMH backend over wmctrl (Cinnamon, XFCE, MATE, plain X11)."""

    @pytest.fixture
    def x11(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "c.toml"))
        from heard import config as cfg
        cfg._cached_load.cache_clear()
        for var in ("HYPRLAND_INSTANCE_SIGNATURE", "SWAYSOCK",
                    "KDE_FULL_SESSION", "WAYLAND_DISPLAY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "X-Cinnamon")
        monkeypatch.setenv("DESKTOP_SESSION", "cinnamon")
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        yield
        cfg._cached_load.cache_clear()

    def test_detect_cinnamon(self, x11):
        assert wm.detect() == "x11"

    def test_detect_display_without_wayland(self, x11, monkeypatch):
        monkeypatch.delenv("XDG_SESSION_TYPE")
        monkeypatch.setenv("DISPLAY", ":0")
        assert wm.detect() == "x11"

    def test_kde_on_x11_still_kde(self, x11, monkeypatch):
        """Richer backends must win over the generic fallback."""
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
        assert wm.detect() == "kde"

    def test_gnome_on_x11_still_gnome(self, x11, monkeypatch):
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
        assert wm.detect() == "gnome"

    # --- parsing ---

    def test_parses_wmctrl_output(self, x11, monkeypatch):
        out = (
            "0x03400004  0 brave-browser.Brave-browser  leo-desktop  Heard - Brave\n"
            "0x04800007  0 nemo.Nemo             leo-desktop chatBotAI\n"
        )
        monkeypatch.setattr(wm, "_run", lambda *a, **k: mock.Mock(stdout=out))
        windows = wm.list_windows()
        assert [w.address for w in windows] == ["0x03400004", "0x04800007"]
        assert windows[0].cls == "Brave-browser"
        assert windows[0].title == "Heard - Brave"     # spaces kept
        assert windows[1].cls == "Nemo"

    def test_empty_title_is_still_a_window(self, x11, monkeypatch):
        out = "0x03000003  0 nemo-desktop.Nemo-desktop  leo-desktop\n"
        monkeypatch.setattr(wm, "_run", lambda *a, **k: mock.Mock(stdout=out))
        windows = wm.list_windows()
        assert len(windows) == 1
        assert windows[0].title == ""

    def test_sticky_window_kept(self, x11, monkeypatch):
        out = "0x05000011 -1 whatsie.WhatSie  leo-desktop  WhatSie\n"
        monkeypatch.setattr(wm, "_run", lambda *a, **k: mock.Mock(stdout=out))
        assert len(wm.list_windows()) == 1

    def test_short_rows_and_blanks_skipped(self, x11, monkeypatch):
        out = "garbage\n\n0x1 0 a.A host title\n"
        monkeypatch.setattr(wm, "_run", lambda *a, **k: mock.Mock(stdout=out))
        assert [w.address for w in wm.list_windows()] == ["0x1"]

    def test_class_without_dot(self, x11, monkeypatch):
        out = "0x1 0 solo host title\n"
        monkeypatch.setattr(wm, "_run", lambda *a, **k: mock.Mock(stdout=out))
        assert wm.list_windows()[0].cls == "solo"

    # --- commands ---

    def test_close_active(self, x11, monkeypatch):
        calls = []
        monkeypatch.setattr(wm, "_run", lambda cmd, **k: calls.append(cmd))
        wm.close_active()
        assert calls == [["wmctrl", "-c", ":ACTIVE:"]]

    def test_toggle_fullscreen(self, x11, monkeypatch):
        calls = []
        monkeypatch.setattr(wm, "_run", lambda cmd, **k: calls.append(cmd))
        wm.toggle_fullscreen()
        assert calls == [["wmctrl", "-r", ":ACTIVE:", "-b", "toggle,fullscreen"]]

    def test_focus_address(self, x11, monkeypatch):
        calls = []
        monkeypatch.setattr(wm, "_run", lambda cmd, **k: calls.append(cmd))
        wm.focus_address("0x03400004")
        assert calls == [["wmctrl", "-i", "-a", "0x03400004"]]

    def test_switch_workspace_is_zero_indexed(self, x11, monkeypatch):
        """heard speaks 1-indexed workspaces; wmctrl desktops start at 0."""
        calls = []
        monkeypatch.setattr(wm, "_run", lambda cmd, **k: calls.append(cmd))
        wm.switch_workspace(3)
        assert calls == [["wmctrl", "-s", "2"]]

    def test_focus_by_token_uses_generic_path(self, x11, monkeypatch):
        out = "0x1 0 brave.Brave host Some Page - Brave\n"
        calls = []

        def fake_run(cmd, **k):
            calls.append(cmd)
            return mock.Mock(stdout=out)

        monkeypatch.setattr(wm, "_run", fake_run)
        wm.focus_by_token("brave")
        assert calls[-1] == ["wmctrl", "-i", "-a", "0x1"]


class TestUnsupportedBackend:
    def test_unknown_backend_raises(self, monkeypatch):
        """A backend with no branch must fail loudly, not fall through to KDE."""
        monkeypatch.setattr(wm, "require", lambda: "beos")
        for fn in (wm.list_windows, wm.close_active, wm.toggle_fullscreen):
            with pytest.raises(RuntimeError, match="unsupported wm backend"):
                fn()
        with pytest.raises(RuntimeError, match="unsupported wm backend"):
            wm.focus_address("0x1")
        with pytest.raises(RuntimeError, match="unsupported wm backend"):
            wm.switch_workspace(1)
        with pytest.raises(RuntimeError, match="unsupported wm backend"):
            wm.minimize_active()


class TestMinimize:
    def test_x11_dispatches_to_helper(self, monkeypatch):
        monkeypatch.setattr(wm, "require", lambda: "x11")
        calls = []
        monkeypatch.setattr(wm, "_x11_minimize", lambda: calls.append(1))
        wm.minimize_active()
        assert calls == [1]

    @pytest.mark.parametrize("backend", ["hyprland", "sway", "kde", "gnome"])
    def test_other_backends_are_unsupported(self, monkeypatch, backend):
        """Minimize is honest about absence: no fake fallback to another action."""
        monkeypatch.setattr(wm, "require", lambda: backend)
        with pytest.raises(RuntimeError, match="unsupported wm backend"):
            wm.minimize_active()

    def test_x11_sends_iconify_to_active_window(self, monkeypatch):
        """Wire shape: WM_CHANGE_STATE=IconicState to the _NET_ACTIVE_WINDOW id."""
        import sys
        import types

        sent = {}

        class FakeRoot:
            def get_full_property(self, atom, _type):
                assert atom == "_NET_ACTIVE_WINDOW"
                return mock.Mock(value=[0x42])

            def send_event(self, msg, event_mask=None):
                sent.update(msg=msg, mask=event_mask)

        class FakeDisplay:
            def screen(self):
                d = types.SimpleNamespace(root=FakeRoot())
                return d

            def intern_atom(self, name):
                return name

            def flush(self):
                pass

            def close(self):
                sent["closed"] = True

        xlib = types.ModuleType("Xlib")
        xmod = types.ModuleType("Xlib.X")
        xmod.AnyPropertyType = 0
        xmod.SubstructureRedirectMask = 1 << 20
        xmod.SubstructureNotifyMask = 1 << 19
        dispmod = types.ModuleType("Xlib.display")
        dispmod.Display = FakeDisplay
        protomod = types.ModuleType("Xlib.protocol")
        evmod = types.ModuleType("Xlib.protocol.event")

        def client_message(**kwargs):
            sent.update(kwargs)
            return kwargs

        evmod.ClientMessage = client_message
        xlib.X, xlib.display = xmod, dispmod
        protomod.event = evmod
        for name, mod in [("Xlib", xlib), ("Xlib.X", xmod),
                          ("Xlib.display", dispmod),
                          ("Xlib.protocol", protomod),
                          ("Xlib.protocol.event", evmod)]:
            monkeypatch.setitem(sys.modules, name, mod)

        wm._x11_minimize()

        assert sent["window"] == 0x42
        assert sent["client_type"] == "WM_CHANGE_STATE"
        assert sent["data"] == (32, [3, 0, 0, 0, 0])
        assert sent["mask"] == (1 << 20) | (1 << 19)
        assert sent["closed"] is True

    def test_x11_without_active_window_raises(self, monkeypatch):
        import sys
        import types

        class FakeRoot:
            def get_full_property(self, _atom, _type):
                return None

        class FakeDisplay:
            def screen(self):
                return types.SimpleNamespace(root=FakeRoot())

            def intern_atom(self, name):
                return name

            def close(self):
                pass

        xlib = types.ModuleType("Xlib")
        xmod = types.ModuleType("Xlib.X")
        xmod.AnyPropertyType = 0
        dispmod = types.ModuleType("Xlib.display")
        dispmod.Display = FakeDisplay
        protomod = types.ModuleType("Xlib.protocol")
        evmod = types.ModuleType("Xlib.protocol.event")
        evmod.ClientMessage = mock.Mock()
        xlib.X, xlib.display = xmod, dispmod
        protomod.event = evmod
        for name, mod in [("Xlib", xlib), ("Xlib.X", xmod),
                          ("Xlib.display", dispmod),
                          ("Xlib.protocol", protomod),
                          ("Xlib.protocol.event", evmod)]:
            monkeypatch.setitem(sys.modules, name, mod)

        with pytest.raises(RuntimeError, match="no active window"):
            wm._x11_minimize()
