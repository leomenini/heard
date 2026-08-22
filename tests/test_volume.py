from unittest import mock

from heard.tools.types import Failed, Ok, Rejected
from heard.tools.volume import volume_control


class TestVolumeControl:
    @mock.patch("heard.tools.volume.subprocess.run")
    def test_up(self, mock_run):
        r = volume_control("up")
        assert isinstance(r, Ok)
        mock_run.assert_called_once()

    @mock.patch("heard.tools.volume.subprocess.run")
    def test_down(self, mock_run):
        r = volume_control("down")
        assert isinstance(r, Ok)
        mock_run.assert_called_once()

    @mock.patch("heard.tools.volume.subprocess.run")
    def test_mute(self, mock_run):
        r = volume_control("mute")
        assert isinstance(r, Ok)
        mock_run.assert_called_once()

    @mock.patch("heard.tools.volume.subprocess.run")
    def test_set(self, mock_run):
        r = volume_control("set", amount="50")
        assert isinstance(r, Ok)
        mock_run.assert_called_once()

    def test_set_missing_amount(self):
        r = volume_control("set")
        assert isinstance(r, Rejected)
        assert r.kind == "invalid_value"

    def test_bad_action(self):
        r = volume_control("shutdown")
        assert isinstance(r, Rejected)
        assert r.kind == "invalid_value"

    @mock.patch("heard.tools.volume.subprocess.run", side_effect=FileNotFoundError("wpctl"))
    def test_missing_wpctl(self, mock_run):
        r = volume_control("up")
        assert isinstance(r, Failed)


class TestUnmute:
    @mock.patch("heard.tools.volume.subprocess.run")
    def test_unmute_sets_mute_zero(self, mock_run):
        r = volume_control("unmute")
        assert isinstance(r, Ok)
        assert "set-mute" in mock_run.call_args[0][0]
        assert "0" in mock_run.call_args[0][0]

    @mock.patch("heard.tools.volume.subprocess.run")
    def test_mute_sets_mute_one(self, mock_run):
        volume_control("mute")
        assert "1" in mock_run.call_args[0][0]

    @mock.patch("heard.tools.volume.subprocess.run")
    def test_amount_dropped_for_non_set(self, mock_run):
        volume_control("mute", amount="sounds")     # junk slot from model
        cmd = mock_run.call_args[0][0]
        assert "%" not in " ".join(cmd)

    @mock.patch("heard.tools.volume.subprocess.run")
    def test_set_rejects_non_numeric(self, mock_run):
        r = volume_control("set", amount="sounds")
        assert isinstance(r, Rejected)
        assert r.kind == "invalid_value"
        mock_run.assert_not_called()
