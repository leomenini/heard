"""Finetune Needle's contrastive head on heard's bilingual prototype bank.

Freezes the encoder; trains contrastive_hidden / contrastive_proj / log_temp
(~82k params) with a class-anchored CLIP loss: utterance embeddings must
match their class-description embedding against in-batch negatives.

Usage:
  uv run python scripts/train_needle_head.py \
      [--checkpoint checkpoints/needle_checkpoint.pkl] \
      [--out checkpoints/needle_head_finetuned.pkl] [--steps 600]

After training, point the daemon at the result:
  export HEARD_CHECKPOINT=checkpoints/needle_head_finetuned.pkl
  HEARD_EMBEDDER=needle uv run python scripts/benchmark_latency.py --only intent
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heard.classifier import PROTOTYPES


def build_dataset():
    """(utterances, labels, class_texts) from the prototype bank."""
    classes = [k for k, v in PROTOTYPES.items() if v[2]]
    class_texts = []
    utts: list[str] = []
    labels: list[int] = []
    for ci, key in enumerate(classes):
        tool, static_args, texts = PROTOTYPES[key]
        arg_words = " ".join(str(v) for v in static_args.values())
        class_texts.append(f"{tool} {arg_words}".strip())
        for t in texts:
            utts.append(t)
            labels.append(ci)
    return classes, class_texts, utts, np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/needle_checkpoint.pkl")
    ap.add_argument("--out", default="checkpoints/needle_head_finetuned.pkl")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import optax
    from needle import SimpleAttentionNetwork, get_tokenizer, load_checkpoint
    from needle.model.architecture import make_padding_mask

    from heard.embedder import NEEDLE_MAX_TOKENS, pad_token_ids

    print(f"loading {args.checkpoint} ...", flush=True)
    params, config = load_checkpoint(args.checkpoint)
    with open(args.checkpoint, "rb") as f:
        raw_cfg = pickle.load(f)["config"]  # noqa: S301
    tok = get_tokenizer()
    model = SimpleAttentionNetwork(config)

    classes, _class_texts, utts, labels = build_dataset()
    n_utts = len(utts)
    print(f"{len(classes)} classes, {n_utts} utterances")

    # fixed-shape source matrix; padded with pad_token_id so needle's
    # internal padding mask excludes filler positions
    rows = pad_token_ids(utts, tok, max_tokens=NEEDLE_MAX_TOKENS)
    src = jnp.array(rows)
    pad_id = tok.pad_token_id

    # Cache encoder pooled features once; the head is downstream, so we only
    # need to run the frozen encoder a single time.
    src_mask = make_padding_mask(src, pad_id)
    enc_out, enc_mask = model.apply(
        {"params": params}, src, src_mask=src_mask, deterministic=True, method="encode_text"
    )
    pooled = model.apply({"params": params}, enc_out, enc_mask, method="_mean_pool")
    pooled = jax.lax.stop_gradient(pooled).astype(jnp.float32)

    head_keys = ("contrastive_hidden", "contrastive_proj", "log_temp")
    frozen = params
    head0 = {k: params[k] for k in head_keys}

    # The shipped contrastive head has decayed to exact zeros, which kills
    # gradients through the L2-normalization layer. Reinitialize it.
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    head0 = {
        "contrastive_hidden": {
            "kernel": jax.random.normal(
                k1, head0["contrastive_hidden"]["kernel"].shape, jnp.float32
            )
            * 0.02,
            "bias": jnp.zeros(head0["contrastive_hidden"]["bias"].shape, jnp.float32),
        },
        "contrastive_proj": {
            "kernel": jax.random.normal(k2, head0["contrastive_proj"]["kernel"].shape, jnp.float32)
            * 0.02,
        },
        "log_temp": jnp.zeros((), jnp.float32),
    }

    def head_forward(pooled_in, head):
        h = jax.nn.relu(
            pooled_in @ head["contrastive_hidden"]["kernel"].astype(jnp.float32)
            + head["contrastive_hidden"]["bias"].astype(jnp.float32)
        )
        projected = h @ head["contrastive_proj"]["kernel"].astype(jnp.float32)
        denom = jnp.sqrt(jnp.sum(projected**2, axis=-1, keepdims=True) + 1e-12)
        return projected / denom

    def embed(head):
        return head_forward(pooled, head)

    def loss_fn(head):
        emb = embed(head)
        temp = jnp.exp(jnp.clip(head["log_temp"], -jnp.log(100.0), jnp.log(100.0)))
        sims = (emb @ emb.T).astype(jnp.float32) * temp
        # exclude self-similarity
        n = sims.shape[0]
        row_idx = jnp.arange(n)
        same_label = labels[:, None] == labels[None, :]
        not_self = row_idx[:, None] != row_idx[None, :]
        mask_same = same_label & not_self
        mask_other = (~same_label) & not_self
        # numerically stable supervised contrastive: log(Σ exp(sim/τ)) per row
        numer = jax.nn.logsumexp(jnp.where(mask_same, sims, -1e9), axis=1)
        denom = jax.nn.logsumexp(jnp.where(mask_same | mask_other, sims, -1e9), axis=1)
        # classes with a single utterance have no positives; mask them out
        has_pos = mask_same.sum(axis=1) > 0
        losses = jnp.where(has_pos, denom - numer, 0.0)
        return jnp.sum(losses) / jnp.maximum(has_pos.sum(), 1)

    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_val = jax.jit(loss_fn)

    def loo_accuracy(head):
        """Leave-one-out nearest-centroid accuracy over prototypes."""
        emb = np.asarray(embed(head), dtype=np.float32)
        correct = 0
        for i in range(n_utts):
            best_c, best_s = None, -2.0
            for c in range(len(classes)):
                idx = [j for j in range(n_utts) if labels[j] == c and j != i]
                if not idx:
                    continue
                v = emb[idx].mean(axis=0)
                v /= max(np.linalg.norm(v), 1e-12)
                s = float(emb[i] @ v)
                if s > best_s:
                    best_c, best_s = c, s
            correct += int(best_c == labels[i])
        return correct / n_utts

    print("\neval BEFORE training:")
    t0 = time.perf_counter()
    acc0 = loo_accuracy(head0)
    print(f"  LOO accuracy: {acc0:.3f}  ({time.perf_counter() - t0:.1f}s)")

    opt = optax.adam(2e-3)
    opt_state = opt.init(head0)
    head = head0

    print(f"\ntraining {args.steps} steps ...")
    for step in range(args.steps):
        grads = grad_fn(head)
        updates, opt_state = opt.update(grads, opt_state)
        head = optax.apply_updates(head, updates)
        if step % 100 == 0 or step == args.steps - 1:
            print(f"  step {step:>4}  loss {float(loss_val(head)):.4f}")

    print("\neval AFTER training:")
    acc1 = loo_accuracy(head)
    print(f"  LOO accuracy: {acc1:.3f}")

    merged = dict(frozen)
    for k in head_keys:
        merged[k] = head[k]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"params": merged, "config": raw_cfg}, f)
    print(f"\nsaved {out}")
    print(
        "try it:\n"
        f"  HEARD_CHECKPOINT={out} HEARD_EMBEDDER=needle "
        "uv run python scripts/benchmark_latency.py --only intent"
    )


if __name__ == "__main__":
    main()
