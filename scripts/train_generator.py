"""CLI entry point for training IndividuateNet (cluster / torch).

Wires: frozen encoders -> IndividuateNet generator -> IndividuationReward ->
GeneratorTrainer. This is a scaffold: the data iterator that yields per-identity
context features + attribute targets must be filled in for the real dataset.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from individuatenet.utils import load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Train IndividuateNet (cluster).")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, default=None,
                    help="checkpoint dir (use a durable $HOME path on the cluster)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))

    import torch

    from individuatenet.models.encoders import FrozenEncoders
    from individuatenet.models.generator import GeneratorConfig, IndividuateNet
    from individuatenet.train.losses import IndividuationReward, RewardWeights
    from individuatenet.train.train_generator import GeneratorTrainer, TrainConfig

    device = cfg.get("device", "cpu")
    encoders = FrozenEncoders(
        backbone=cfg.encoders.backbone,
        pretrained=cfg.encoders.pretrained,
        device=device,
        offline=cfg.encoders.get("offline", False),
    )

    gen_cfg = GeneratorConfig(
        base_lm=cfg.generator.base_lm,
        prefix_tokens=cfg.generator.prefix_tokens,
        context_dim=cfg.generator.context_dim,
        max_new_tokens=cfg.generator.max_new_tokens,
        offline=cfg.encoders.get("offline", False),
    )
    model = IndividuateNet(gen_cfg)
    model = model.to(device)  # warm-start runs before GeneratorTrainer's .to(device)

    reward = IndividuationReward(
        RewardWeights(
            lambda_attr=cfg.generator.lambda_attr,
            beta_fluency=cfg.generator.beta_fluency,
            temperature=cfg.u.temperature,
        )
    )

    def encode_text(captions):
        return torch.as_tensor(encoders.encode_text(captions), dtype=torch.float32)

    # ---- real per-identity batching -------------------------------------
    import numpy as np

    from individuatenet.data.real import load_dataset
    from individuatenet.data.captions import build_captions, _ATTR_PHRASES

    ds = load_dataset(cfg)
    by_id: dict = {}
    for s in ds:
        by_id.setdefault(s.identity, []).append(s)
    ids = [i for i, v in by_id.items() if len(v) >= cfg.get("min_imgs", 3)]

    rng = np.random.default_rng(cfg.get("seed", 0))
    rng.shuffle(ids)
    max_train_ids = cfg.get("max_train_ids", 500)
    if max_train_ids:
        ids = ids[:max_train_ids]
    if not ids:
        raise RuntimeError("no identities with enough images; check data.root")

    attr_keys: list = []
    for i in ids:
        cand = [k for k in _ATTR_PHRASES if k in by_id[i][0].attributes]
        if cand:
            attr_keys = cand
            break
    if not attr_keys:
        attr_keys = ["Male"]
    n_attr = len(attr_keys)

    # per-identity context (= mean image embedding, train-time only) + modal attrs
    ctx_rows, attr_rows, baseline_caps = [], [], []
    for i in ids:
        imgs = by_id[i]
        emb = encoders.encode_image([s.image_path for s in imgs]).mean(axis=0)
        modal = {
            k: (1 if sum(int(s.attributes.get(k, 0)) for s in imgs) * 2 >= len(imgs) else 0)
            for k in attr_keys
        }
        ctx_rows.append(emb)
        attr_rows.append([modal[k] for k in attr_keys])
        baseline_caps.append(build_captions(modal, "attribute"))
    context_mat = np.asarray(ctx_rows, dtype=np.float32)   # (N, context_dim)
    attr_mat = np.asarray(attr_rows, dtype=np.float32)     # (N, A)

    # frozen attribute predictor q(a|t): pretrain on baseline captions, then freeze.
    # Supplies the reward's attribute-leakage term (returns per-attribute logits).
    class AttrPredictor(torch.nn.Module):
        def __init__(self, dim_t: int, dim_a: int, hidden: int = 256):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim_t, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, dim_a),
            )

        def forward(self, t):
            return self.net(t)

    attr_net = AttrPredictor(encoders.dim, n_attr).to(device)
    base_text = encode_text(baseline_caps).to(device)
    base_attr = torch.as_tensor(attr_mat, dtype=torch.float32, device=device)
    opt_a = torch.optim.Adam(attr_net.parameters(), lr=1e-3)
    attr_net.train()
    for _ in range(cfg.get("attr_pretrain_steps", 500)):
        opt_a.zero_grad()
        loss_a = torch.nn.functional.binary_cross_entropy_with_logits(
            attr_net(base_text), base_attr
        )
        loss_a.backward()
        opt_a.step()
    attr_net.eval()
    for p in attr_net.parameters():
        p.requires_grad_(False)

    n_ids = len(ids)
    batch_size = cfg.generator.batch_size

    def data_iter():
        sel = rng.integers(0, n_ids, size=batch_size)
        ctx = torch.as_tensor(context_mat[sel], dtype=torch.float32, device=device)
        return {
            "context": ctx,                         # (B, context_dim) soft-prompt source
            "id_anchor": ctx.clone(),               # (B, D) identity anchor for I(T;Y)
            "attr": torch.as_tensor(attr_mat[sel], dtype=torch.float32, device=device),
            "attr_net": attr_net,                   # frozen q(a|t) for the leak penalty
        }

    import os

    ckpt_dir = args.out or cfg.get("output_dir", "results/celeba")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "generator.pt")

    # ---- supervised warm-start -------------------------------------------
    # A random prefix makes GPT-2 emit generic, repetitive text; REINFORCE from
    # that cold start rarely converges. Prefix-tune (LM loss) to reproduce the
    # baseline captions first, giving RL a fluent, on-topic initialisation.
    # Skipped when resuming (a checkpoint already exists).
    warmup = cfg.generator.get("warmup_steps", 300)
    if warmup and not os.path.exists(ckpt_path):
        tok = model.tokenizer
        enc = tok(baseline_caps, return_tensors="pt", padding=True,
                  truncation=True, max_length=cfg.generator.max_new_tokens)
        w_ids = enc["input_ids"].to(device)
        w_attn = enc["attention_mask"].to(device)
        w_ctx = torch.as_tensor(context_mat, dtype=torch.float32, device=device)
        w_opt = torch.optim.Adam(model.context.parameters(), lr=cfg.generator.lr)
        model.train()
        n_w = w_ids.shape[0]
        for it in range(warmup):
            sel = torch.as_tensor(
                rng.integers(0, n_w, size=min(batch_size, n_w)), device=device
            )
            out = model(w_ctx[sel], w_ids[sel], attention_mask=w_attn[sel])
            w_opt.zero_grad()
            out.loss.backward()
            w_opt.step()
            if it % 50 == 0:
                print(f"[warmup {it:4d}] lm_loss={float(out.loss):.3f}")
        model.eval()

    trainer = GeneratorTrainer(
        model, reward, encode_text,
        TrainConfig(lr=cfg.generator.lr, steps=cfg.generator.steps,
                    batch_size=cfg.generator.batch_size,
                    max_new_tokens=cfg.generator.max_new_tokens,
                    ckpt_path=ckpt_path,
                    save_every=cfg.generator.get("save_every", 500)),
        device=device,
    )
    trainer.fit(data_iter)
    print(f"[done] generator checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
