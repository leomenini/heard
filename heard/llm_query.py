"""Query mode: ask a configured LLM and speak the answer.

Triggered when a transcript starts with "question" or "pregunta". The rest
of the transcript is sent to an OpenAI-compatible chat endpoint; the returned
text is spoken via TTS and printed.
"""

import json
import os
import subprocess
import tempfile
import threading
import urllib.request

QUERY_TRIGGERS = ("question", "pregunta")


def is_query(text: str) -> tuple[bool, str]:
    """Return (True, prompt_without_trigger) if text starts with a trigger word."""
    stripped = text.strip()
    lowered = stripped.lower()
    for trigger in QUERY_TRIGGERS:
        if lowered.startswith(trigger):
            rest = stripped[len(trigger):].strip()
            return True, rest
    return False, stripped


def _config_value(key: str) -> str | bool:
    from . import config

    return config.cached(key)


def _llm_payload(prompt: str) -> bytes:
    body: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
    }
    model = str(_config_value("llm_model") or "")
    if model:
        body["model"] = model
    return json.dumps(body).encode("utf-8")


def query(prompt: str) -> str:
    """Send the prompt to the configured LLM and return the answer text."""
    url = str(_config_value("llm_url") or "")
    if not url:
        return "LLM not configured: set llm_url (and llm_api_key / llm_model) in config."

    key = str(_config_value("llm_api_key") or "")
    req = urllib.request.Request(  # noqa: S310
        url,
        data=_llm_payload(prompt),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError):
        return "LLM response did not contain expected chat completion fields."
    except Exception as e:  # noqa: BLE001
        return f"LLM error: {e}"


def _play_file(path: str) -> None:
    """Best-effort playback of an audio file; raises on total failure."""
    for player in (["mpv", "--no-video", "--really-quiet", path],
                   ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]):
        try:
            subprocess.run(player, check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("no audio player found")


def _speak_blocking(text: str) -> None:
    """Speak text out loud, falling back to console output if TTS fails."""
    if not _config_value("tts_enabled"):
        print(f"  heard (tts disabled): {text}")
        return

    lang = str(_config_value("language") or "en")

    # Primary: online TTS via gTTS (covers en and es well).
    try:
        from gtts import gTTS

        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.write_to_fp(f)
            path = f.name
        try:
            _play_file(path)
        finally:
            os.remove(path)
        return
    except Exception:  # noqa: BLE001
        pass

    # Fallback for English: local flite + aplay.
    if lang == "en":
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            try:
                subprocess.run(["flite", "-t", text, "-o", path], check=True)
                subprocess.run(["aplay", "-q", path], check=True)
            finally:
                os.remove(path)
            return
        except Exception:  # noqa: BLE001
            pass

    # Last resort: just show the text.
    print(f"  heard (tts unavailable): {text}")


def speak(text: str) -> None:
    """Speak text without blocking the daemon's main loop."""
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()
