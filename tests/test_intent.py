"""Intent module tests -- Needle is mocked in conftest.py."""

import pytest


@pytest.fixture(autouse=True)
def _clear_config_cache():
    from heard import config

    config._cached_load.cache_clear()
    yield
    config._cached_load.cache_clear()


class TestIntent:
    def test_module_imports(self):
        import heard.intent
        assert heard.intent.MAX_GEN_LEN == 64
        assert heard.intent.CHECKPOINT_PATH == "checkpoints/needle_checkpoint.pkl"

    def test_resolution_dataclass(self):
        from heard.intent import Resolution
        r = Resolution(raw_query="test", tool_call=None, reason="declined", latency_ms=None)
        assert r.raw_query == "test"
        assert r.tool_call is None
        assert r.reason == "declined"

    def test_resolution_with_tool_call(self):
        from heard.intent import Resolution
        tc = {"name": "volume_control", "arguments": {"action": "up"}}
        r = Resolution(raw_query="turn up", tool_call=tc, reason=None, latency_ms=100.0)
        assert r.tool_call == tc
        assert r.latency_ms == 100.0

    def test_tools_json_cache_exists(self):
        import json

        from heard.tools.registry import known_tools
        # Verify the function exists and produces valid JSON
        expected = json.dumps(known_tools())
        assert expected  # non-empty


class TestCheckpointResolution:
    def test_missing_checkpoint_raises_with_instructions(self, tmp_path, monkeypatch):
        import pytest

        from heard import intent

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HEARD_CHECKPOINT", raising=False)
        with pytest.raises(RuntimeError, match="HEARD_CHECKPOINT"):
            intent._checkpoint_file()

    def test_env_var_wins(self, tmp_path, monkeypatch):
        from heard import intent

        target = tmp_path / "weights.pkl"
        target.write_bytes(b"x")
        monkeypatch.setenv("HEARD_CHECKPOINT", str(target))
        assert intent._checkpoint_file() == str(target)

    def test_config_checkpoint_wins_over_default(self, tmp_path, monkeypatch):
        from heard import config

        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "heard.toml"
        monkeypatch.setenv("HEARD_CONFIG", str(cfg))
        target = tmp_path / "custom.pkl"
        target.write_bytes(b"x")
        cfg.write_text(f'checkpoint = "{target}"\n')
        config._cached_load.cache_clear()

        from heard import intent

        monkeypatch.delenv("HEARD_CHECKPOINT", raising=False)
        assert intent._checkpoint_file() == str(target)

    def test_env_wins_over_config_checkpoint(self, tmp_path, monkeypatch):
        from heard import config

        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / "heard.toml"
        monkeypatch.setenv("HEARD_CONFIG", str(cfg))
        env_target = tmp_path / "env.pkl"
        env_target.write_bytes(b"x")
        cfg_target = tmp_path / "cfg.pkl"
        cfg_target.write_bytes(b"x")
        cfg.write_text(f'checkpoint = "{cfg_target}"\n')
        config._cached_load.cache_clear()
        monkeypatch.setenv("HEARD_CHECKPOINT", str(env_target))

        from heard import intent

        assert intent._checkpoint_file() == str(env_target)

    def test_glob_fallback(self, tmp_path, monkeypatch):
        from heard import intent

        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "checkpoint_epoch3.pkl").write_bytes(b"x")
        monkeypatch.delenv("HEARD_CHECKPOINT", raising=False)
        resolved = intent._checkpoint_file()
        assert resolved.endswith("checkpoint_epoch3.pkl")

    def test_default_path_preferred_over_glob(self, tmp_path, monkeypatch):
        from pathlib import Path

        from heard import intent

        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        (ckpt / "needle_checkpoint.pkl").write_bytes(b"x")
        (ckpt / "other.pkl").write_bytes(b"x")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HEARD_CHECKPOINT", raising=False)
        resolved = Path(intent._checkpoint_file()).resolve()
        assert resolved == (ckpt / "needle_checkpoint.pkl").resolve()
