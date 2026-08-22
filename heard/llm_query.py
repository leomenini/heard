"""Query mode: ask a configured LLM and speak the answer.

Triggered when a transcript starts with "question" or "pregunta". The rest
of the transcript is sent to an OpenAI-compatible chat endpoint; the returned
text is spoken via TTS and printed.

Speech output is local-first. Auto chain: piper (ONNX, voice downloaded once)
-> gTTS when the optional `tts` extra is installed -> flite (English only).
Config `tts_backend` pins one backend or disables speech ("none").
"""

import contextlib
import json
import os
import subprocess
import tempfile
import threading
import urllib.request
import wave
from pathlib import Path

QUERY_TRIGGERS = ("question", "pregunta")

# Piper voices per language; ~60MB each, downloaded once into the cache.
PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "es": "es_ES-sharvard-medium",
}
_PIPER_VOICE_URL = ("https://huggingface.co/rhasspy/piper-voices/resolve/"
                    "v1.0.0/{prefix}/{locale}/{name}/{quality}/{voice}{ext}")


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


def _piper_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "heard" / "piper"


def _piper_voice_url(voice: str, ext: str) -> str:
    locale, rest = voice.split("-", 1)
    name, quality = rest.rsplit("-", 1)
    return _PIPER_VOICE_URL.format(prefix=locale.split("_")[0], locale=locale,
                                   name=name, quality=quality,
                                   voice=voice, ext=ext)


def ensure_piper_voice(lang: str) -> tuple[Path, Path]:
    """Model + config paths under the cache, downloading either once."""
    voice_name = PIPER_VOICES.get(lang, PIPER_VOICES["en"])
    cache = _piper_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    model = cache / f"{voice_name}.onnx"
    config = cache / f"{voice_name}.onnx.json"
    for ext, target in ((".onnx", model), (".onnx.json", config)):
        if target.is_file():
            continue
        url = _piper_voice_url(voice_name, ext)
        print(f"heard: downloading piper voice {voice_name}{ext} (once)")
        partial = target.with_suffix(ext + ".part")
        urllib.request.urlretrieve(url, partial)  # noqa: S310 -- https only
        os.replace(partial, target)  # noqa: S310
    return model, config


def _speak_piper(text: str, lang: str) -> None:
    """Local ONNX synthesis via piper; downloads its voice on first use."""
    from piper import PiperVoice

    model, _config = ensure_piper_voice(lang)
    voice_obj = PiperVoice.load(model)
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(wav_path, "wb") as wav:
            voice_obj.synthesize_wav(text, wav)
        _play_file(wav_path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(wav_path)


def _speak_gtts(text: str, lang: str) -> None:
    """Online gTTS; requires the optional [tts] extra."""
    from gtts import gTTS

    tts = gTTS(text=text, lang=lang)
    fd, path = tempfile.mkstemp(suffix=".mp3")
    with os.fdopen(fd, "wb") as f:
        tts.write_to_fp(f)
    try:
        _play_file(path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def _speak_flite(text: str, lang: str) -> None:
    """Local flite + aplay fallback (English only)."""
    if lang != "en":
        raise RuntimeError("flite is English-only")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(["flite", "-t", text, "-o", path], check=True)
        subprocess.run(["aplay", "-q", path], check=True)
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


BACKENDS = {
    "piper": _speak_piper,
    "gtts": _speak_gtts,
    "flite": _speak_flite,
}


def _backend_chain() -> list[str]:
    """Configured backend if pinned, else the local-first auto chain."""
    configured = str(_config_value("tts_backend") or "").strip().lower()
    if configured == "none":
        return []
    if configured in BACKENDS:
        return [configured]
    return ["piper", "gtts", "flite"]


def _speak_blocking(text: str) -> None:
    """Speak text out loud, falling back to console output if TTS fails."""
    if not _config_value("tts_enabled"):
        print(f"  heard (tts disabled): {text}")
        return

    lang = str(_config_value("language") or "en").lower()
    for backend in _backend_chain():
        try:
            BACKENDS[backend](text, lang)
            return
        except Exception:  # noqa: BLE001 -- next link in the chain
            continue

    # Last resort: just show the text.
    print(f"  heard (tts unavailable): {text}")


def speak(text: str) -> None:
    """Speak text without blocking the daemon's main loop."""
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()
