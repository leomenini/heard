import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from needle import (
    SimpleAttentionNetwork,
    generate,
    get_tokenizer,
    load_checkpoint,
)

from .classifier import Classifier
from .embedder import resolve_encoder
from .tools import registry

MAX_GEN_LEN = 64

CHECKPOINT_PATH = "checkpoints/needle_checkpoint.pkl"


def _checkpoint_file() -> str:
    """Resolve the Needle weights: $HEARD_CHECKPOINT, config checkpoint,
    default path, then any *.pkl in checkpoints/. Raises with instructions
    when nothing is found."""
    candidates: list[str] = []
    env_ckpt = os.environ.get("HEARD_CHECKPOINT")
    if env_ckpt:
        candidates.append(env_ckpt)
    try:
        from . import config

        cfg_ckpt = str(config.cached("checkpoint") or "")
        if cfg_ckpt:
            candidates.append(cfg_ckpt)
    except Exception:
        cfg_ckpt = ""
    candidates.append(CHECKPOINT_PATH)
    ckpt_dir = Path("checkpoints")
    if ckpt_dir.is_dir():
        candidates += sorted(str(p) for p in ckpt_dir.glob("*.pkl"))
    for c in candidates:
        if Path(c).is_file():
            return c
    raise RuntimeError(
        "Needle checkpoint not found.\n"
        f"Looked for: {CHECKPOINT_PATH}, any *.pkl in checkpoints/, "
        f"config 'checkpoint' ({cfg_ckpt or 'not set'}), "
        f"and $HEARD_CHECKPOINT ({env_ckpt or 'not set'})"
        "\nDrop the finetuned weights into checkpoints/ or set:\n"
        "  heard config set checkpoint checkpoints/needle_head_finetuned.pkl\n"
        "or:\n"
        "  export HEARD_CHECKPOINT=checkpoints/needle_head_finetuned.pkl"
    )


@dataclass(frozen=True)
class Resolution:
    raw_query: str
    tool_call: dict | None
    reason: str | None
    latency_ms: float | None
    source: str = "generative"   # fast | generative
    score: float | None = None


@lru_cache(maxsize=1)
def _model():
    """Load once. ~1s load + ~8s JIT on first generate()"""

    params, config = load_checkpoint(_checkpoint_file())
    return SimpleAttentionNetwork(config), params, get_tokenizer()

@lru_cache(maxsize=1)
def _tools_json() -> str:
    # minified: this string is re-encoded on every generative fallback
    return json.dumps(registry.known_tools(), separators=(",", ":"))


@lru_cache(maxsize=1)
def _encoder():
    """Embedding backend for the fast path (needle or fastembed MiniLM)."""
    encode = resolve_encoder(needle_triple=_model())
    # Copy any backend-specific thresholds to the stable _encode function so
    # the classifier picks them up regardless of caching wrappers.
    for name in ("accept_score", "margin", "floor_decline"):
        if hasattr(encode, name):
            setattr(_encode, name, getattr(encode, name))
    return encode


def _encode(texts: list[str]) -> np.ndarray:
    return _encoder()(texts)


@lru_cache(maxsize=1)
def _classifier() -> Classifier:
    """Build centroids. Encodes ~120 short utterances once at warmup."""
    return Classifier(_encode)


def _generate_resolve(query: str, t0: float) -> Resolution:
    model, params, tok = _model()
    raw = generate(model, params, tok, query=query,
                   tools=_tools_json(), stream=False, max_gen_len=MAX_GEN_LEN)
    ms = (time.perf_counter() - t0) * 1000

    try:
        calls = json.loads(raw)
    except ValueError:
        return Resolution(query, None, "unparseable", ms)
    if not calls:
        return Resolution(query, None, "declined", ms)    # [] is correct output
    return Resolution(query, calls[0], None, ms)          # single-shot by design


def resolve(query: str) -> Resolution:
    t0 = time.perf_counter()
    verdict = _classifier().match(query)

    if verdict.tool_call is not None or verdict.declined:
        ms = (time.perf_counter() - t0) * 1000
        reason = "declined" if verdict.declined else None
        return Resolution(query, verdict.tool_call, reason, ms,
                          source="fast", score=verdict.score)

    return _generate_resolve(query, t0)


def warmup() -> Resolution:
    """Load weights, JIT both encoder and decoder paths."""
    verdict = _classifier().match("warmup probe")
    if verdict.tool_call is None and not verdict.declined:
        return _generate_resolve("test", time.perf_counter())
    return Resolution("warmup probe", verdict.tool_call, None, 0.0, source="fast")
