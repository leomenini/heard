from unittest import mock

from heard.tools.media import media_control, _mpris_send, _playerctl
from heard.tools.types import Ok, Rejected, Failed


class TestMediaControl:
    @mock.patch("heard.tools.media._mpris_send", return_value=True)
    def test_play_via_mpris(self, mock_send):
        r = media_control("play")
        assert isinstance(r, Ok)
        assert "mpris" in r.detail
        mock_send.assert_called_once_with("play")

    @mock.patch("heard.tools.media._mpris_send", return_value=False)
    @mock.patch("heard.tools.media.subprocess.run")
    def test_fallback_to_playerctl(self, mock_run, mock_send):
        r = media_control("pause")
        assert isinstance(r, Ok)
        mock_run.assert_called_once()

    @mock.patch("heard.tools.media._mpris_send", return_value=True)
    def test_all_actions_map(self, mock_send):
        for action in ("play", "pause", "next", "previous"):
            assert isinstance(media_control(action), Ok)

    def test_bad_action(self):
        r = media_control("shutdown")
        assert isinstance(r, Rejected)
        assert r.kind == "invalid_value"

    @mock.patch("heard.tools.media._mpris_send", return_value=False)
    @mock.patch("heard.tools.media.subprocess.run", side_effect=FileNotFoundError("playerctl"))
    def test_missing_playerctl(self, mock_run, mock_send):
        r = media_control("play")
        assert isinstance(r, Failed)

    @mock.patch("heard.tools.media._mpris_send", return_value=False)
    @mock.patch("heard.tools.media.subprocess.run",
                side_effect=mock.Mock(side_effect=__import__("subprocess").CalledProcessError(3, "playerctl")))
    def test_playerctl_nonzero_exit(self, mock_run, mock_send):
        r = media_control("next")
        assert isinstance(r, Failed)


class TestMprisSend:
    @mock.patch("heard.tools.media._players", side_effect=OSError("no bus"))
    def test_no_bus_returns_false(self, mock_players):
        assert _mpris_send("play") is False

    @mock.patch("heard.tools.media._players", return_value=[])
    def test_no_players_returns_false(self, mock_players):
        assert _mpris_send("play") is False


class TestPlayerctl:
    @mock.patch("heard.tools.media.subprocess.run")
    def test_ok(self, mock_run):
        r = _playerctl("play")
        assert isinstance(r, Ok)
