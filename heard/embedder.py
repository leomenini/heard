"""Embedding backends for the fast-path classifier.

Needle's contrastive head only produces useful vectors on checkpoints where
the retrieval stage was actually trained; pretrain-only checkpoints decay it
to zero. HEARD_EMBEDDER picks the backend explicitly; "auto" (default) probes
needle first and falls back to a local ONNX MiniLM when the space is dead.
"""

import os

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def degenerate(embs) -> bool:
    """True when embeddings carry no signal (all-zero / near-zero rows)."""
    arr = np.asarray(embs)
    if arr.size == 0:
        return True
    return float(np.abs(arr).max()) < 1e-6


class NeedleEmbedder:
    """encode_for_retrieval over the already-loaded generative weights."""

    name = "needle"

    def __init__(self, model, params, tokenizer):
        self._model, self._params, self._tok = model, params, tokenizer

    def encode(self, texts: list[str]) -> np.ndarray:
        from needle import encode_for_retrieval

        return encode_for_retrieval(self._model, self._params, self._tok,
                                    list(texts))


class FastEmbedder:
    """ONNX int8 MiniLM via fastembed. Model downloads once to ~/.cache."""

    name = "fastembed"

    def __init__(self):
        from fastembed import TextEmbedding

        print(f"heard: loading {EMBED_MODEL} (onnx, cached after first run)")
        self._model = TextEmbedding(EMBED_MODEL)

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
    needle/fastembed force one backend.
    needle_triple: (model, params, tokenizer) shared with generative intent.
    """
    mode = os.environ.get("HEARD_EMBEDDER", "auto").lower()

    if mode in ("auto", "needle") and needle_triple is not None:
        emb = NeedleEmbedder(*needle_triple)
        if mode == "needle":
            return emb.encode
        if not degenerate(_probe(emb.encode)):
            return emb.encode
        if mode == "needle":
            raise RuntimeError(
                "HEARD_EMBEDDER=needle but this checkpoint's contrastive head "
                "is degenerate (zero vectors). Finetune the retrieval stage or "
                "use HEARD_EMBEDDER=fastembed.")
        print("heard: needle contrastive head is untrained in this checkpoint; "
              "using fastembed MiniLM for the fast path")

    if mode not in ("auto", "fastembed"):
        raise ValueError(f"unknown HEARD_EMBEDDER mode {mode!r}")
    return FastEmbedder().encode
