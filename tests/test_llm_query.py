"""Query mode: LLM + TTS."""

import json
from unittest import mock

import pytest

from heard import llm_query


@pytest.fixture
def cfg(monkeypatch):
    values = {
        "llm_url": "",
        "llm_api_key": "",
        "llm_model": "",
        "language": "en",
        "tts_enabled": True,
        "tts_backend": "",
    }
    monkeypatch.setattr("heard.config.cached", lambda key: values.get(key, ""))
    return values


class TestIsQuery:
    def test_question_trigger(self):
        ok, prompt = llm_query.is_query("Question what is the weather")
        assert ok
        assert prompt == "what is the weather"

    def test_pregunta_trigger(self):
        ok, prompt = llm_query.is_query("Pregunta qué hora es")
        assert ok
        assert prompt == "qué hora es"

    def test_no_trigger(self):
        ok, prompt = llm_query.is_query("turn the volume up")
        assert not ok
        assert prompt == "turn the volume up"

    def test_trigger_alone(self):
        ok, prompt = llm_query.is_query("question")
        assert ok
        assert prompt == ""


class TestQuery:
    def test_unconfigured_returns_message(self, cfg):
        assert "LLM not configured" in llm_query.query("hello")

    def test_success_parses_chat_completion(self, cfg):
        cfg["llm_url"] = "https://api.example.com/v1/chat/completions"
        cfg["llm_api_key"] = "secret"
        cfg["llm_model"] = "gpt-4o-mini"

        fake_resp = mock.Mock()
        fake_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": " 42 "}}],
        }).encode()
        fake_resp.__enter__ = mock.Mock(return_value=fake_resp)
        fake_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=fake_resp) as uo:
            answer = llm_query.query("what is the answer")

        assert answer == "42"
        req = uo.call_args[0][0]
        assert req.full_url == "https://api.example.com/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer secret"
        body = json.loads(req.data)
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"][0]["content"] == "what is the answer"

    def test_missing_fields_returns_message(self, cfg):
        cfg["llm_url"] = "https://api.example.com/v1/chat/completions"

        fake_resp = mock.Mock()
        fake_resp.read.return_value = json.dumps({"choices": []}).encode()
        fake_resp.__enter__ = mock.Mock(return_value=fake_resp)
        fake_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=fake_resp):
            assert "did not contain" in llm_query.query("hello")


class TestSpeak:
    def test_disabled_prints_instead(self, cfg, capsys):
        cfg["tts_enabled"] = False
        llm_query._speak_blocking("hello")  # blocking variant for test
        captured = capsys.readouterr()
        assert "hello" in captured.out


class TestBackendChain:
    def test_auto_is_local_first(self, cfg):
        assert llm_query._backend_chain() == ["piper", "gtts", "flite"]

    def test_pinned_backend(self, cfg):
        cfg["tts_backend"] = "flite"
        assert llm_query._backend_chain() == ["flite"]

    def test_none_disables(self, cfg):
        cfg["tts_backend"] = "none"
        assert llm_query._backend_chain() == []

    def test_first_working_backend_wins(self, cfg):
        calls = []
        backends = {
            "piper": lambda t, lang: (_ for _ in ()).throw(RuntimeError("down")),
            "gtts": lambda t, lang: calls.append("gtts"),
            "flite": lambda t, lang: calls.append("flite"),
        }
        with mock.patch.dict(llm_query.BACKENDS, backends):
            llm_query._speak_blocking("hi")
        assert calls == ["gtts"]

    def test_all_fail_prints_text(self, cfg, capsys):
        backends = {k: (lambda t, lang: (_ for _ in ()).throw(RuntimeError("x")))
                    for k in llm_query.BACKENDS}
        with mock.patch.dict(llm_query.BACKENDS, backends):
            llm_query._speak_blocking("hello")
        assert "tts unavailable" in capsys.readouterr().out

    def test_pinned_backend_skips_earlier_links(self, cfg):
        calls = []

        def flite(text, lang):
            calls.append("flite")

        def boom(text, lang):
            raise AssertionError("piper must not run when pinned to flite")

        cfg["tts_backend"] = "flite"
        with mock.patch.dict(llm_query.BACKENDS,
                             {"piper": boom, "gtts": boom, "flite": flite}):
            llm_query._speak_blocking("hi")
        assert calls == ["flite"]


class TestPiperVoice:
    def test_voice_url_shape(self):
        url = llm_query._piper_voice_url("en_US-lessac-medium", ".onnx")
        assert url == ("https://huggingface.co/rhasspy/piper-voices/resolve/"
                       "v1.0.0/en/en_US/lessac/medium/"
                       "en_US-lessac-medium.onnx")

    def test_es_voice_url(self):
        url = llm_query._piper_voice_url("es_ES-sharvard-medium", ".onnx.json")
        assert "/es/es_ES/sharvard/medium/" in url
        assert url.endswith("es_ES-sharvard-medium.onnx.json")

    def test_existing_files_not_redownloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_query, "_piper_cache_dir", lambda: tmp_path)
        voice_name = llm_query.PIPER_VOICES["en"]
        (tmp_path / f"{voice_name}.onnx").write_bytes(b"x")
        (tmp_path / f"{voice_name}.onnx.json").write_text("{}")

        def fail(*a, **k):
            raise AssertionError("must not download existing files")

        with mock.patch.object(llm_query.urllib.request, "urlretrieve", fail):
            model, config = llm_query.ensure_piper_voice("en")
        assert model.read_bytes() == b"x"

    def test_missing_files_download(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llm_query, "_piper_cache_dir", lambda: tmp_path)
        fetched = []

        def fake_retrieve(url, target):
            fetched.append(url)
            target.write_bytes(b"d")

        with mock.patch.object(llm_query.urllib.request, "urlretrieve",
                               fake_retrieve):
            model, config = llm_query.ensure_piper_voice("es")
        assert len(fetched) == 2
        assert all(u.endswith((".onnx", ".onnx.json")) for u in fetched)
        assert model.is_file() and config.is_file()
        assert not list(tmp_path.glob("*.part"))
