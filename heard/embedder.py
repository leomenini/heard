"""Embedding backends for the fast-path classifier.

Needle's contrastive head only produces useful vectors on checkpoints where
the retrieval stage was actually trained; pretrain-only checkpoints decay it
to zero. HEARD_EMBEDDER picks the backend explicitly; "auto" (default) probes
needle first and falls back to a local ONNX MiniLM when the space is dead.
"""

import os

import numpy as np

# en: fast English-only model. es: cross-lingual alignment so Spanish
# utterances land near their English prototypes.
EMBED_MODELS = {
    "en": "sentence-transformers/all-MiniLM-L6-v2",
    "es": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
DEFAULT_LANGUAGE = "en"


def model_for(language: str | None = None) -> str:
    if env := os.environ.get("HEARD_EMBED_MODEL"):
        return env
    try:
        from . import config

        configured = str(config.cached("embedder_model") or "")
    except Exception:
        configured = ""
    if configured:
        return configured
    lang = (language or DEFAULT_LANGUAGE).lower()
    return EMBED_MODELS.get(lang, EMBED_MODELS["en"])


def degenerate(embs) -> bool:
    """True when embeddings carry no signal (all-zero / near-zero rows)."""
    arr = np.asarray(embs)
    if arr.size == 0:
        return True
    return float(np.abs(arr).max()) < 1e-6


# --- fast jitted needle encode ---------------------------------------------
NEEDLE_MAX_TOKENS = 32
_fast_encode_cache: dict = {}


def pad_token_ids(texts: list[str], tokenizer,
                  max_tokens: int = NEEDLE_MAX_TOKENS) -> np.ndarray:
    """(B, max_tokens) int32 matrix, left-aligned, padded with pad_token_id."""
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = tokenizer.eos_token_id
    rows = []
    for text in texts:
        ids = list(tokenizer.encode(text))[:max_tokens]
        ids += [pad] * (max_tokens - len(ids))
        rows.append(ids)
    return np.array(rows, dtype=np.int32)


def fast_encode(model, params, tokenizer, texts: list[str]) -> np.ndarray:
    """JIT'd encode_contrastive; L2-normalized (B, 128).

    Compiled per exact batch size so runtime queries (almost always B=1)
    don't pay the FLOPs of a larger padded batch, and padded positions use
    the tokenizer's pad id so encode_contrastive's internal mask ignores them.
    """
    import jax
    import jax.numpy as jnp

    n = len(texts)
    if n == 0:
        return np.empty((0, 128), dtype=np.float32)

    key = (id(model), n)
    if key not in _fast_encode_cache:
        def forward(src):
            # encode_contrastive builds its own padding mask from pad ids
            return model.apply({"params": params}, src,
                               deterministic=True,
                               method="encode_contrastive")

        _fast_encode_cache[key] = {
            "fn": jax.jit(forward),
            "pad": lambda t: pad_token_ids(t, tokenizer),
        }
    entry = _fast_encode_cache[key]

    rows = entry["pad"](texts)
    out = np.asarray(entry["fn"](jnp.array(rows)), dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


class NeedleEmbedder:
    """encode_for_retrieval over the already-loaded generative weights.

    Uses a JIT-compiled, fixed-shape forward (queries padded to
    NEEDLE_MAX_TOKENS): eager JAX dispatch costs >1s per query on CPU,
    the compiled path single-digit ms.
    """

    name = "needle"

    def __init__(self, model, params, tokenizer):
        self._model, self._params, self._tok = model, params, tokenizer

    def encode(self, texts: list[str]) -> np.ndarray:
        return fast_encode(self._model, self._params, self._tok,
                           list(texts))


class FastEmbedder:
    """ONNX int8 sentence encoder via fastembed. Downloads once to ~/.cache."""

    name = "fastembed"

    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name or model_for()
        print(f"heard: loading {self.model_name} (onnx, cached after first run)")
        self._model = TextEmbedding(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.array(list(self._model.embed(list(texts))), dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


def _probe(encode) -> np.ndarray:
    return encode(["how's my battery", "play some music",
                   "what is the weather tomorrow"])


def resolve_encoder(needle_triple=None):
    """Return an encode(texts)->np.ndarray callable per HEARD_EMBEDDER.

    auto: use needle when its contrastive space is alive, else fastembed.
    needle/fastembed force one backend. The fastembed model follows config
    `language` (es -> multilingual) unless HEARD_EMBED_MODEL overrides.
    needle_triple: (model, params, tokenizer) shared with generative intent.
    """
    mode = os.environ.get("HEARD_EMBEDDER", "auto").lower()

    if mode in ("auto", "needle") and needle_triple is not None:
        emb = NeedleEmbedder(*needle_triple)

        def encode(texts):
            return emb.encode(texts)

        if not degenerate(_probe(encode)):
            # Tuned on scripts/benchmark_latency.py fast-path probe:
            # needle's finetuned head is high-confidence but can over-accept
            # off-topic queries, so it needs a higher floor than fastembed.
            encode.accept_score = 0.85  # type: ignore[attr-defined]
            return encode
        if mode == "needle":
            raise RuntimeError(
                "HEARD_EMBEDDER=needle but this checkpoint's contrastive head "
                "is degenerate (zero vectors). Finetune the retrieval stage or "
                "use HEARD_EMBEDDER=fastembed.")
        print("heard: needle contrastive head is untrained in this checkpoint; "
              "using fastembed for the fast path")

    if mode not in ("auto", "fastembed"):
        raise ValueError(f"unknown HEARD_EMBEDDER mode {mode!r}")
    try:
        language = str(config_language())
    except Exception:
        language = DEFAULT_LANGUAGE
    return FastEmbedder(model_for(language)).encode


def config_language() -> str | None:
    from . import config

    value = str(config.cached("language") or "")
    return value or None
