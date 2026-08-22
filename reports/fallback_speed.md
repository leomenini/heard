# Fallback speed: investigation notes

Goal: take the generative fallback tail from ~6.5s to hundreds of
milliseconds on a laptop CPU. This document records what the spike found,
what to run, and the decision framework. Nothing here is wired into
production paths.

## Where the time actually goes

`needle.generate()` re-forwards the **entire decoder prefix for every
token** (`decode_fn(params, dec_buffer, encoder_out, enc_mask)` inside the
token loop — see `needle/model/run.py`). There is no KV cache, so decode
cost grows roughly quadratically over the sequence:

```
total ≈ encode(prefill) + Σ_{i=1..n} forward(i tokens)
      ≈ encode + n²/2 × per-token-forward-cost(at 1 token)
```

That structure means the biggest win is algorithmic (caching), not numeric
(quantization). Quantizing an O(n²) loop still leaves O(n²).

Run the measurement:

```bash
uv run python scripts/export_fallback.py            # baseline + step profile
uv run python scripts/export_fallback.py --onnx     # + export & int8 attempt
```

## Options considered, in attack order

### 1. KV cache in the decoder step (primary recommendation)

Patch needle's `generate()` to carry `K/V` projections between steps and
forward only the newest token each iteration. Expected effect at n≈40
generated tokens: decode drops from ~n×(average full-prefix forward) to
~n×(single-token forward) — an order of magnitude if the step profile shows
the predicted growth curve.

- Effort: medium — touches `needle/model/run.py`; JIT shapes become
  per-step unless the cache is padded to `max_gen_len`.
- Risk: numerics must stay identical (verify outputs token-by-token against
  the un-cached path).
- Kill criteria: if the step profile shows flat per-step cost (no growth),
  caching buys little and this line of work stops.

### 2. ONNX export + int8 dynamic quantization of the decoder step

`scripts/export_fallback.py --onnx` exports just the decoder-step function
(fixed shapes) via jax2onnx or jax2tf+tf2onnx, then quantizes with ORT.
This stacks with #1: cache-aware single-token steps exported to ONNX int8
is the "hundreds of ms" end state.

- Dependencies: none in-tree by design. The tf2onnx route needs
  `tensorflow-cpu tf2onnx onnx onnxruntime` installed ad hoc.
- Kill criteria: custom ops failing export, or int8 output divergence that
  breaks constrained decoding (schema-prefix logits must stay argmax-stable).

### 3. GGUF / llama.cpp port (last resort)

Needle's architecture is small (26M) but custom; llama.cpp needs a new arch
implementation upstream. Highest effort, worst maintenance story. Only worth
revisiting if #1+#2 stall AND fallback traffic grows beyond ~15%.

## Results

Fill from `scripts/export_fallback.py` runs on the i5-1335U:

| Measurement | Value | Notes |
|---|---|---|
| baseline p50 | _pending_ | `Play some music`, post-warmup |
| encode prefill | _pending_ | tools JSON included |
| decode forwards (n) | _pending_ | until EOS |
| first-third ms/step | _pending_ | |
| last-third ms/step | _pending_ | growth x = KV-cache payoff estimate |
| JAX step p50/p95 | _pending_ | fixed early-step shape |
| ONNX fp32 step p50/p95 | _pending_ | |
| ONNX int8 step p50/p95 | _pending_ | |

## Decision

Pending hardware numbers. If the step-profile growth factor lands ≥3x and
the cached path validates token-identically, implement #1 behind
`HEARD_INTENT_BACKEND=jax-kv` and keep the current path as fallback;
revisit #2 once #1's shapes stabilize.
