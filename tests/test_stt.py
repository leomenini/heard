"""STT module tests -- all hardware access is mocked in conftest.py."""

from unittest import mock

import numpy as np
import pytest

from heard import stt

KEY_LEFTSHIFT = 42   # matches the mocked ecodes in conftest
KEY_POWER = 116


class FakeDev:
    def __init__(self, name, keys, raises=None):
        self.name = name
        self._keys = keys
        self._raises = raises

    def capabilities(self):
        if self._raises:
            raise self._raises
        return {1: list(self._keys)} if self._keys else {}   # 1 == EV_KEY


def _patch_devices(monkeypatch, devs):
    paths = [f"/dev/input/event{i}" for i in range(len(devs))]
    monkeypatch.setattr(stt, "list_devices", lambda: paths)
    monkeypatch.setattr(stt, "InputDevice", lambda p: devs[int(p[-1])])
    return devs


class TestSTT:
    def test_module_imports(self):
        assert stt.SAMPLE_RATE == 16000

    def test_keycode(self):
        assert stt._keycode("KEY_LEFTSHIFT") == 42
        assert stt._keycode("KEY_LEFTCONTROL") == 29

    def test_keycode_invalid(self):
        with pytest.raises(ValueError, match="KEY_NONEXISTENT"):
            stt._keycode("KEY_NONEXISTENT")


class TestFindInputDevices:
    def test_no_devices(self, monkeypatch):
        monkeypatch.setattr(stt, "list_devices", list)
        with pytest.raises(RuntimeError, match="input group|/dev/input"):
            stt._find_input_devices(KEY_LEFTSHIFT, "KEY_LEFTSHIFT")

    def test_selects_by_keycode_not_ev_key(self, monkeypatch):
        # 'Power Button' has EV_KEY but never emits Shift; must be skipped
        power = FakeDev("Power Button", [KEY_POWER])
        keyboard = FakeDev("AT Translated Set 2 keyboard", [KEY_LEFTSHIFT])
        _patch_devices(monkeypatch, [power, keyboard])
        matched = stt._find_input_devices(KEY_LEFTSHIFT, "KEY_LEFTSHIFT")
        assert [d.name for d in matched] == ["AT Translated Set 2 keyboard"]

    def test_no_device_advertises_key(self, monkeypatch):
        _patch_devices(monkeypatch, [FakeDev("Power Button", [KEY_POWER])])
        with pytest.raises(RuntimeError, match="advertises KEY_LEFTSHIFT"):
            stt._find_input_devices(KEY_LEFTSHIFT, "KEY_LEFTSHIFT")

    def test_permission_denied_with_match(self, monkeypatch):
        locked = FakeDev("Secret Keyboard", [KEY_LEFTSHIFT],
                         raises=PermissionError(13, "nope"))
        _patch_devices(monkeypatch, [locked])
        with pytest.raises(RuntimeError, match="usermod"):
            stt._find_input_devices(KEY_LEFTSHIFT, "KEY_LEFTSHIFT")

    def test_multiple_matches_all_returned(self, monkeypatch):
        a = FakeDev("AT keyboard", [KEY_LEFTSHIFT])
        b = FakeDev("USB keyboard", [KEY_LEFTSHIFT])
        _patch_devices(monkeypatch, [a, b])
        assert len(stt._find_input_devices(KEY_LEFTSHIFT, "KEY_LEFTSHIFT")) == 2

    def test_capability_error_skipped(self, monkeypatch):
        broken = FakeDev("Broken", [], raises=OSError("device vanished"))
        good = FakeDev("Good", [KEY_LEFTSHIFT])
        _patch_devices(monkeypatch, [broken, good])
        matched = stt._find_input_devices(KEY_LEFTSHIFT, "KEY_LEFTSHIFT")
        assert [d.name for d in matched] == ["Good"]


class TestLanguageConfig:
    def test_default_language_en(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "none.toml"))
        from heard import config as cfg
        cfg._cached_load.cache_clear()
        try:
            assert stt._stt_language() == "en"
        finally:
            cfg._cached_load.cache_clear()

    def test_language_es_flows_to_transcribe(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "c.toml"))
        (tmp_path / "c.toml").write_text('language = "es"\n')
        from heard import config as cfg
        cfg._cached_load.cache_clear()

        fake = mock.Mock()
        fake.transcribe.return_value = (iter([]), None)
        monkeypatch.setattr(stt, "_model", lambda: fake)
        monkeypatch.setattr(stt, "_speech_bounds", lambda a: (0, len(a)))
        audio = np.zeros(stt.SAMPLE_RATE, dtype=np.float32)

        try:
            stt.transcribe(audio)
            kwargs = fake.transcribe.call_args.kwargs
            assert kwargs["language"] == "es"
        finally:
            cfg._cached_load.cache_clear()

    def test_model_size_from_config(self, monkeypatch, tmp_path):
        import sys
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "m.toml"))
        (tmp_path / "m.toml").write_text('stt_model_size = "tiny"\n')
        from heard import config as cfg
        cfg._cached_load.cache_clear()
        stt._model.cache_clear()
        try:
            stt._model()
            args, _ = sys.modules["faster_whisper"].WhisperModel.call_args
            assert args[0] == "tiny"
        finally:
            stt._model.cache_clear()
            cfg._cached_load.cache_clear()

