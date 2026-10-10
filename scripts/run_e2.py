"""E2: the training-free U predicts zero-shot outcomes.

We build a grid of regimes varying individuation ``p`` and per-identity image
count, estimate ``U`` (training-free, caption-side) and separately *measure*
zero-shot AUC/EER via enrolment. We then fit
``AUC = sigma(alpha + beta*U + gamma*log n_img)`` and report the Spearman rank
correlation between ``U`` and AUC and between ``U`` and the invariance gap.

Offline by default (numpy only). Reproduces the shape of Fig. 2 / Table 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from individuatenet.data.synthetic import make_regime, split_context_probe
from individuatenet.eval import zero_shot_enrolment
from individuatenet.eval.predict_u import fit_u_predictor
from individuatenet.mi import estimate_u
from individuatenet.utils import set_seed


def build_grid(args) -> dict:
    p_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    img_counts = [5, 10, 20]
    rows = {"u": [], "auc": [], "eer": [], "n_img": [], "p": [], "imgs": []}

    for imgs in img_counts:
        for p in p_levels:
            us, aucs, eers = [], [], []
            for seed in range(args.seeds):
                set_seed(seed)
                reg = make_regime(
                    n_ids=args.n_ids, imgs_per_id=imgs,
                    dim=args.dim, attr_dim=args.attr_dim, seed=seed,
                )
                # training-free U from per-image captions at individuation p
                text_pi = reg.caption_embeddings(p, per_image=True)
                est = estimate_u(text_pi, reg.image_emb, reg.id_labels, reg.attr,
                                 backend="numpy", seed=seed)
                # measured enrolment outcome from per-identity captions at p
                _, probe_idx = split_context_probe(reg, test_frac=args.test_frac, seed=seed)
                enrol_text = reg.caption_embeddings(p, per_image=False)
                res = zero_shot_enrolment(
                    enrol_text, reg.ids, reg.image_emb[probe_idx],
                    reg.id_labels[probe_idx], seed=seed,
                )
                us.append(est.u)
                aucs.append(res.auc)
                eers.append(res.eer)
            rows["u"].append(float(np.mean(us)))
            rows["auc"].append(float(np.mean(aucs)))
            rows["eer"].append(float(np.mean(eers)))
            rows["n_img"].append(imgs)
            rows["p"].append(p)
            rows["imgs"].append(imgs)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="E2 U-predicts-outcome analysis.")
    ap.add_argument("--synthetic", action="store_true", default=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_ids", type=int, default=150)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--attr_dim", type=int, default=16)
    ap.add_argument("--test_frac", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="results/e2_synthetic.json")
    args = ap.parse_args()

    grid = build_grid(args)
    fit = fit_u_predictor(
        grid["u"], grid["auc"], grid["n_img"], gap_values=grid["eer"]
    )

    print("\nE2 -- U predicts zero-shot enrolment (synthetic regime)\n")
    print(f"  Spearman(U, AUC)        = {fit.spearman_u_auc:+.3f}")
    print(f"  Pearson(U, EER-gap)     = {fit.pearson_u_gap:+.3f}")
    print(f"  logit-AUC fit R^2       = {fit.r2_logit:.3f}")
    print(f"  AUC = sigma({fit.alpha:+.2f} {fit.beta:+.2f}*U {fit.gamma:+.2f}*log n_img)")

    print("\n  p     n_img     U      AUC     EER")
    for p, n, u, a, e in zip(grid["p"], grid["n_img"], grid["u"], grid["auc"], grid["eer"]):
        print(f"  {p:4.2f}  {n:5d}  {u:6.3f}  {a:6.3f}  {e:6.3f}")

    payload = {"grid": grid, "fit": fit.__dict__}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
