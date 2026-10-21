"""Fig. 2: training-free U predicts zero-shot enrolment AUC.

Reads results/e2_real_<dataset>_s<seed>.json (written by run_e2_real.py) and
draws a scatter of U vs measured AUC, coloured by caption source and shaped by
dataset, with a pooled least-squares fit and the Spearman correlation.

Usage:
  python scripts/plot_e2.py \
    --inputs results/e2_real_vggface2_s{0,1,2}.json results/e2_real_celeba_s{0,1,2}.json \
    --out fig2_e2.pdf
(or just --glob 'results/e2_real_*_s*.json')
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC_ORDER = ["discriminative", "attribute", "rich", "generated"]
SRC_LABEL = {
    "discriminative": "discriminative",
    "attribute": "attribute",
    "rich": "rich (BLIP VLM)",
    "generated": "IndividuateNet",
}
SRC_COLOR = {
    "discriminative": "#9e9e9e",
    "attribute": "#1f77b4",
    "rich": "#ff7f0e",
    "generated": "#2ca02c",
}
DS_MARKER = {"vggface2": "o", "celeba": "s"}
DS_LABEL = {"vggface2": "VGGFace2", "celeba": "CelebA"}


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--glob", default="results/e2_real_*_s*.json")
    ap.add_argument("--out", default="fig2_e2.pdf")
    args = ap.parse_args()

    files = args.inputs or sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no E2 json found (glob={args.glob})")

    pts = []  # (dataset, source, u, auc)
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        ds = d.get("dataset", Path(f).stem).lower()
        ds = "vggface2" if "vgg" in ds else ("celeba" if "celeb" in ds else ds)
        for src, p in d["points"].items():
            pts.append((ds, src, p["u"], p["auc"]))

    us = np.array([p[2] for p in pts])
    aucs = np.array([p[3] for p in pts])
    rho = _spearman(us, aucs)
    # pooled linear fit for the guide line
    b, a = np.polyfit(us, aucs, 1)

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    xs = np.linspace(us.min() - 0.03, us.max() + 0.03, 50)
    ax.plot(xs, a + b * xs, color="0.5", lw=1.0, ls="--", zorder=1,
            label=f"fit ($\\rho={rho:+.3f}$)")

    seen = set()
    for ds, src, u, auc in pts:
        lbl = None
        key = ("src", src)
        if key not in seen:
            lbl = SRC_LABEL.get(src, src); seen.add(key)
        ax.scatter(u, auc, s=42, c=SRC_COLOR.get(src, "k"),
                   marker=DS_MARKER.get(ds, "o"), edgecolors="white",
                   linewidths=0.6, zorder=3, label=lbl)

    # dataset-shape legend proxies
    for ds, mk in DS_MARKER.items():
        ax.scatter([], [], marker=mk, c="0.3", s=42, label=DS_LABEL[ds])

    ax.set_xlabel(r"Individuation $\mathcal{U}(T)=I(T;Y)-I(T;A\,|\,Y)$")
    ax.set_ylabel("Zero-shot enrolment AUC")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9, ncol=1)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, bbox_inches="tight")
    print(f"Spearman(U,AUC)={rho:+.3f} over {len(pts)} points -> saved {out}")


if __name__ == "__main__":
    main()
