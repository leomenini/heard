"""User configuration: ~/.config/heard/config.toml (HEARD_CONFIG overrides).

Flat key/value store with typed defaults. `language` is the main switch:
"en" or "es" -- it drives STT decoding and the embedding model choice; the
classifier itself is always bilingual (prototypes carry both languages).
"""

import os
import tomllib
from functools import lru_cache
from pathlib import Path

import tomli_w

DEFAULTS: dict[str, str | bool] = {
    "language": "en",            # en | es
    "ptt_key": "KEY_LEFTSHIFT",
    "checkpoint": "",            # empty = checkpoints/needle_checkpoint.pkl
    "embedder_model": "",        # empty = auto per language
    "stt_model_size": "base",    # tiny | base | small
    "stt_cpu_threads": "",       # empty = ctranslate2 default; try 2 on hybrid CPUs
    "wm_backend": "auto",        # auto | hyprland | sway | kde
    "events": True,              # append local JSONL usage log
    "llm_url": "",               # OpenAI-compatible chat completions endpoint
    "llm_api_key": "",           # Bearer token for llm_url
    "llm_model": "",             # e.g. gpt-4o-mini; required by most endpoints
    "tts_enabled": True,         # speak LLM answers out loud
    "screenrecord_output": "",   # default display output (e.g. DP-1)
    "screenrecord_outputs": "",  # label map: One=DP-1,Two=HDMI-1
    "screenrecord_folder": "~/Videos",
}

VALID_LANGUAGE = ("en", "es")
VALID_WM_BACKEND = ("auto", "hyprland", "sway", "kde")


def config_path() -> Path:
    override = os.environ.get("HEARD_CONFIG")
    if override:
        return Path(override)
    return Path.home() / ".config" / "heard" / "config.toml"


def load() -> dict[str, str | bool]:
    path = config_path()
    raw: dict = {}
    if path.is_file():
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    merged = dict(DEFAULTS)
    for k, v in raw.items():
        if k in DEFAULTS:
            merged[k] = v
    return merged


def get(key: str) -> str | bool:
    cfg = load()
    if key not in DEFAULTS:
        raise KeyError(f"unknown config key {key!r}; valid: {sorted(DEFAULTS)}")
    return cfg[key]


def set_key(key: str, value: str) -> None:
    if key not in DEFAULTS:
        raise KeyError(f"unknown config key {key!r}; valid: {sorted(DEFAULTS)}")
    if key == "language" and value not in VALID_LANGUAGE:
        raise ValueError(f"language must be one of {VALID_LANGUAGE}")
    if key == "wm_backend" and value not in VALID_WM_BACKEND:
        raise ValueError(f"wm_backend must be one of {VALID_WM_BACKEND}")
    path = config_path()
    raw: dict = {}
    if path.is_file():
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    default_type = type(DEFAULTS[key])
    if default_type is bool:
        parsed = value.strip().lower() in ("true", "1", "yes")
    else:
        parsed = value
    raw[key] = parsed
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(raw, f)
    _cached_load.cache_clear()


@lru_cache(maxsize=1)
def _cached_load() -> dict[str, str | bool]:
    return load()


def cached(key: str) -> str | bool:
    """Per-process snapshot for hot paths (STT/embedder read every call)."""
    return _cached_load()[key]
