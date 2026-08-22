from typer.testing import CliRunner

from heard.cli import app

runner = CliRunner()


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "listen" in result.output
        assert "config" in result.output

    def test_listen_help(self):
        result = runner.invoke(app, ["listen", "--help"])
        assert result.exit_code == 0
        assert "hold the PTT key" in result.output

    def test_config_help(self):
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "get" in result.output
        assert "set" in result.output
        assert "show" in result.output

    def test_config_get(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "c.toml"))
        result = runner.invoke(app, ["config", "get", "language"])
        assert result.exit_code == 0
        assert "language = en" in result.output

    def test_config_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "c.toml"))
        result = runner.invoke(app, ["config", "set", "language", "es"])
        assert result.exit_code == 0
        assert "set language = es" in result.output

    def test_config_show(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "c.toml"))
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "ptt_key" in result.output
        assert "stt_model_size" in result.output

    def test_config_invalid_key_fails_cleanly(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "c.toml"))
        result = runner.invoke(app, ["config", "get", "wm.backend"])
        assert result.exit_code == 1
        assert "unknown config key" in result.output
