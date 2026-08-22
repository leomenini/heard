"""Embedding backend helpers."""

import numpy as np

from heard import embedder


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text):
        return [2, 3]


class FakeTokenizerNoPad:
    pad_token_id = None
    eos_token_id = 7

    def encode(self, text):
        return [4]


def test_pad_token_ids_uses_pad_token_id():
    tok = FakeTokenizer()
    out = embedder.pad_token_ids(["hi"], tok, max_tokens=4)
    assert out.shape == (1, 4)
    assert out.dtype == np.int32
    assert list(out[0]) == [2, 3, 0, 0]


def test_pad_token_ids_falls_back_to_eos():
    tok = FakeTokenizerNoPad()
    out = embedder.pad_token_ids(["hi"], tok, max_tokens=4)
    assert list(out[0]) == [4, 7, 7, 7]


def test_pad_token_ids_truncates():
    tok = FakeTokenizer()
    out = embedder.pad_token_ids(["hi"], tok, max_tokens=1)
    assert list(out[0]) == [2]


def test_fast_encode_empty():
    # No model call needed for an empty batch.
    out = embedder.fast_encode(None, None, None, [])
    assert out.shape == (0, 128)
