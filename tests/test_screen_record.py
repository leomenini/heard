"""Screen recording tool."""

from pathlib import Path
from unittest import mock

import pytest

from heard.tools import screen_record
from heard.tools.types import Failed, Ok, Rejected


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    values = {
        "screenrecord_output": "",
        "screenrecord_outputs": "",
        "screenrecord_folder": str(tmp_path),
    }
    monkeypatch.setattr("heard.config.cached", lambda key: values.get(key, ""))
    monkeypatch.setattr(screen_record, "PIDFILE", tmp_path / "screenrecord.pid")
    return values


class TestResolveOutput:
    def test_empty_defaults(self, cfg):
        assert screen_record._resolve_output(None) is None

    def test_label_mapping(self, cfg):
        cfg["screenrecord_outputs"] = "One=DP-1,Two=HDMI-1"
        assert screen_record._resolve_output("One") == "DP-1"
        assert screen_record._resolve_output("two") == "HDMI-1"

    def test_unmapped_label_used_directly(self, cfg):
        assert screen_record._resolve_output("DP-2") == "DP-2"


class TestStart:
    @mock.patch("heard.tools.screen_record.subprocess.Popen")
    def test_start_wf_recorder_default_screen(self, mock_popen, cfg):
        proc = mock.Mock()
        proc.pid = 1234
        proc.poll.return_value = None
        mock_popen.return_value = proc

        r = screen_record.screen_record("start")
        assert isinstance(r, Ok)
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "wf-recorder"
        assert "-f" in cmd
        assert "screenrecord_" in cmd[cmd.index("-f") + 1]

    @mock.patch("heard.tools.screen_record.subprocess.Popen")
    def test_start_specific_output(self, mock_popen, cfg):
        cfg["screenrecord_output"] = "DP-1"
        proc = mock.Mock()
        proc.pid = 1234
        proc.poll.return_value = None
        mock_popen.return_value = proc

        screen_record.screen_record("start")
        cmd = mock_popen.call_args[0][0]
        assert cmd[cmd.index("-o") + 1] == "DP-1"

    @mock.patch("heard.tools.screen_record.subprocess.Popen")
    def test_start_already_recording(self, mock_popen, cfg):
        screen_record._save_state({"pid": 1234, "path": "/tmp/old.mp4"})
        with mock.patch.object(screen_record, "_process_alive", return_value=True):
            r = screen_record.screen_record("start")
        assert isinstance(r, Failed)
        assert "already recording" in r.reason

    @mock.patch("heard.tools.screen_record.subprocess.Popen")
    def test_fallback_to_ffmpeg(self, mock_popen, cfg):
        def side_effect(cmd, **kwargs):
            if cmd[0] == "wf-recorder":
                raise FileNotFoundError(cmd[0])
            proc = mock.Mock()
            proc.pid = 5678
            proc.poll.return_value = None
            return proc

        mock_popen.side_effect = side_effect
        r = screen_record.screen_record("start")
        assert isinstance(r, Ok)
        calls = [c[0][0][0] for c in mock_popen.call_args_list]
        assert calls == ["wf-recorder", "ffmpeg"]


class TestFfmpegCmd:
    def test_grab_target_follows_display_env(self, monkeypatch):
        """x11grab must follow $DISPLAY, not assume :0.0."""
        monkeypatch.setenv("DISPLAY", ":1")
        cmd = screen_record._ffmpeg_cmd(Path("/tmp/x.mp4"))
        assert cmd[cmd.index("-i") + 1] == ":1"

    def test_grab_target_defaults_without_display(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        cmd = screen_record._ffmpeg_cmd(Path("/tmp/x.mp4"))
        assert cmd[cmd.index("-i") + 1] == ":0"


class TestStop:
    def test_stop_when_not_recording(self, cfg):
        r = screen_record.screen_record("stop")
        assert isinstance(r, Failed)
        assert "not recording" in r.reason

    def test_stop_terminates_and_clears(self, cfg):
        screen_record._save_state({"pid": 1234, "path": "/tmp/out.mp4"})
        killed = []

        def fake_kill(pid, sig):
            killed.append((pid, sig))

        with (
            mock.patch.object(screen_record, "_process_alive", return_value=True),
            mock.patch("os.kill", side_effect=fake_kill),
            mock.patch("time.sleep"),
        ):
            r = screen_record.screen_record("stop")

        assert isinstance(r, Ok)
        assert "/tmp/out.mp4" in r.detail
        assert (1234, screen_record.signal.SIGTERM) in killed
        assert not screen_record.PIDFILE.exists()


class TestValidation:
    def test_invalid_action(self, cfg):
        r = screen_record.screen_record("pause")
        assert isinstance(r, Rejected)
        assert r.kind == "invalid_value"
