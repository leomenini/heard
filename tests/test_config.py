import pytest

from heard import config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    path = tmp_path / "heard.toml"
    monkeypatch.setenv("HEARD_CONFIG", str(path))
    config._cached_load.cache_clear()
    yield path
    config._cached_load.cache_clear()


class TestConfigLoad:
    def test_defaults_when_missing(self):
        cfg = config.load()
        assert cfg["language"] == "en"
        assert cfg["ptt_key"] == "KEY_LEFTSHIFT"
        assert cfg["events"] is True

    def test_unknown_keys_ignored(self, isolated_config):
        isolated_config.write_text('language = "es"\nbogus = 1\n')
        config._cached_load.cache_clear()
        cfg = config.load()
        assert cfg["language"] == "es"
        assert "bogus" not in cfg


class TestConfigSet:
    def test_roundtrip(self, isolated_config):
        config.set_key("language", "es")
        config._cached_load.cache_clear()
        assert config.get("language") == "es"

    def test_bool_coercion(self, isolated_config):
        config.set_key("events", "false")
        config._cached_load.cache_clear()
        assert config.get("events") is False

    def test_invalid_language_rejected(self):
        with pytest.raises(ValueError, match="language"):
            config.set_key("language", "fr")

    def test_invalid_key_rejected(self):
        with pytest.raises(KeyError, match="bogus"):
            config.set_key("bogus", "x")

    def test_set_preserves_other_keys(self, isolated_config):
        config.set_key("ptt_key", "KEY_LEFTCONTROL")
        config.set_key("language", "es")
        config._cached_load.cache_clear()
        assert config.get("ptt_key") == "KEY_LEFTCONTROL"
        assert config.get("language") == "es"


class TestCachedSnapshot:
    def test_reflects_set_key(self):
        before = config.cached("language")
        config.set_key("language", "es")
        after = config.cached("language")
        assert before == "en" and after == "es"


class TestCpuThreads:
    def test_default_is_auto(self):
        assert config.get("stt_cpu_threads") == ""

    def test_numeric_roundtrip(self, isolated_config):
        config.set_key("stt_cpu_threads", "2")
        config._cached_load.cache_clear()
        assert config.get("stt_cpu_threads") == "2"
