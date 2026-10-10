"""Reward-scale diagnostic for IndividuateNet training.

Prints the magnitude of each reward term on real data so ``lambda_attr`` /
``beta_fluency`` can be set *before* spending a cluster allocation:

    reward = I_hat(T;Y) - lambda_attr * BCE(a|t) - beta_fluency * KL(policy||base)

The identity term is a per-sample InfoNCE log-prob bounded by ``log(batch)``,
while the attribute BCE is *summed* over ~N attributes and otherwise dwarfs it.
This script reports both and suggests a ``lambda_attr`` that balances the two at
init (both by mean magnitude and by std, since REINFORCE sees the baselined
advantage).

Run on the cluster (needs open_clip weights):

    python scripts/check_reward_scale.py --config configs/celeba.yaml
"""

from __future__ import annotations

import argparse
import math

import _bootstrap  # noqa: F401

import numpy as np

from individuatenet.utils import load_config, set_seed


class _AttrPredictor:  # placeholder for type clarity; real class built in main()
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Reward-scale diagnostic (cluster).")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--max_ids", type=int, default=200)
    ap.add_argument("--min_imgs", type=int, default=4)
    ap.add_argument("--attr_pretrain_steps", type=int, default=300)
    ap.add_argument("--sample_generated", action="store_true",
                    help="also sample from an UNTRAINED generator to preview text")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))

    import torch

    from individuatenet.models.encoders import FrozenEncoders
    from individuatenet.data.real import load_dataset
    from individuatenet.data.captions import build_captions, _ATTR_PHRASES
    from individuatenet.train.losses import IndividuationReward, RewardWeights

    device = cfg.get("device", "cpu")
    encoders = FrozenEncoders(
        backbone=cfg.encoders.backbone,
        pretrained=cfg.encoders.pretrained,
        device=device,
        offline=cfg.encoders.get("offline", False),
    )

    ds = load_dataset(cfg)
    by_id: dict = {}
    for s in ds:
        by_id.setdefault(s.identity, []).append(s)
    rng = np.random.default_rng(cfg.get("seed", 0))
    ids = [i for i, v in by_id.items() if len(v) >= args.min_imgs]
    ids = [ids[k] for k in rng.permutation(len(ids))][: args.max_ids]
    if not ids:
        raise RuntimeError("no identities with enough images; check data.root")

    attr_keys: list = []
    for i in ids:
        cand = [k for k in _ATTR_PHRASES if k in by_id[i][0].attributes]
        if cand:
            attr_keys = cand
            break
    attr_keys = attr_keys or ["Male"]
    n_attr = len(attr_keys)

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
    context_mat = np.asarray(ctx_rows, dtype=np.float32)
    attr_mat = np.asarray(attr_rows, dtype=np.float32)
    n = len(ids)

    # frozen attribute predictor q(a|t), trained on the baseline captions
    class AttrPredictor(torch.nn.Module):
        def __init__(self, dt: int, da: int, h: int = 256):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dt, h), torch.nn.ReLU(), torch.nn.Linear(h, da)
            )

        def forward(self, t):
            return self.net(t)

    attr_net = AttrPredictor(encoders.dim, n_attr).to(device)
    base_text = torch.as_tensor(
        encoders.encode_text(baseline_caps), dtype=torch.float32, device=device
    )
    base_attr = torch.as_tensor(attr_mat, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(attr_net.parameters(), lr=1e-3)
    for _ in range(args.attr_pretrain_steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            attr_net(base_text), base_attr
        )
        loss.backward()
        opt.step()
    attr_net.eval()
    for p in attr_net.parameters():
        p.requires_grad_(False)

    reward = IndividuationReward(
        RewardWeights(
            lambda_attr=cfg.generator.lambda_attr,
            beta_fluency=cfg.generator.beta_fluency,
            temperature=cfg.u.get("temperature", 0.07),
        )
    )

    ctx = torch.as_tensor(context_mat, dtype=torch.float32, device=device)
    with torch.no_grad():
        i_ty = reward.identity_term(base_text, ctx)                       # (n,)
        bce = -reward.attribute_term(base_text, attr_net(base_text), base_attr)  # (n,)

    i_ty_m, i_ty_s = float(i_ty.mean()), float(i_ty.std())
    bce_m, bce_s = float(bce.mean()), float(bce.std())
    chance = -math.log(n)

    print("\n=== reward-scale diagnostic (baseline attribute captions) ===")
    print(f"identities={n}  attributes={n_attr}  temperature={reward.w.temperature}")
    print(f"I(T;Y) identity term : mean={i_ty_m:+.3f}  std={i_ty_s:.3f}  "
          f"(chance={chance:.3f}, max=0)")
    print(f"attr-leak BCE (sum)  : mean={bce_m:.3f}  std={bce_s:.3f}  "
          f"(random~={n_attr * math.log(2):.3f})")
    print(f"current lambda_attr={reward.w.lambda_attr}: "
          f"penalty={reward.w.lambda_attr * bce_m:.3f} vs |i_ty|={abs(i_ty_m):.3f}")
    lam_mean = abs(i_ty_m) / max(bce_m, 1e-6)
    lam_std = i_ty_s / max(bce_s, 1e-6)
    print(f"[suggest] lambda_attr ~= {lam_mean:.3f} (balance mean magnitude), "
          f"or ~= {lam_std:.3f} (balance std / advantage sensitivity)")
    print("          set it in configs/celeba.yaml -> generator.lambda_attr\n")

    if args.sample_generated:
        from individuatenet.models.generator import GeneratorConfig, IndividuateNet
        gcfg = GeneratorConfig(
            base_lm=cfg.generator.base_lm,
            prefix_tokens=cfg.generator.prefix_tokens,
            context_dim=cfg.generator.context_dim,
            max_new_tokens=cfg.generator.max_new_tokens,
            offline=cfg.encoders.get("offline", False),
        )
        gmodel = IndividuateNet(gcfg).to(device)
        gmodel.eval()
        k = min(6, n)
        tok = gmodel.generate(ctx[:k], gcfg.max_new_tokens, greedy=True)
        caps = gmodel.decode(tok)
        print("--- UNTRAINED generator caption preview (fluency sanity) ---")
        for j, c in enumerate(caps):
            print(f"  [{j}] {' '.join(c.split())[:120]!r}")
        print()


if __name__ == "__main__":
    main()
