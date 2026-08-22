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
