"""Fig. 3 (inset): the attribute-penalty weight lambda trades grounding for AUC.

Reads the E5 ablation outputs for the lambda sweep --- noattr (lambda=0), full
(lambda=0.5), lam1.0, lam1.5 --- and plots zero-shot AUC of the generated
source against lambda, annotating the over-suppression collapse. Each point is
the ``generated`` AUC from scripts/run_e1.py on CelebA.

Usage:
  python scripts/plot_lambda_sweep.py --dir results --prefix e1_e5_ --out fig3_lambda.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# variant file stem -> lambda value on the x-axis
SWEEP = [
    ("noattr", 0.0),
    ("full", 0.5),
    ("lam1.0", 1.0),
    ("lam1.5", 1.5),
]


def _auc(path: Path, src: str = "generated") -> float:
    d = json.loads(path.read_text(encoding="utf-8"))
    m = d.get(src, {})
    a = m.get("AUC", [float("nan")])
    return a[0] if isinstance(a, (list, tuple)) else a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--prefix", default="e1_e5_")
    ap.add_argument("--out", default="fig3_lambda.pdf")
    args = ap.parse_args()
    root = Path(args.dir)

    xs, ys = [], []
    for stem, lam in SWEEP:
        f = root / f"{args.prefix}{stem}.json"
        if not f.exists():
            print(f"[skip] missing {f.name}")
            continue
        xs.append(lam)
        ys.append(_auc(f))
    if not xs:
        raise SystemExit("no lambda-sweep JSONs found")

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot(xs, ys, "-o", color="#2ca02c", lw=1.8, ms=6, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7)
    # mark the winning operating point
    if 0.5 in xs:
        iy = ys[xs.index(0.5)]
        ax.scatter([0.5], [iy], s=130, facecolors="none",
                   edgecolors="#d62728", lw=1.6, zorder=4, label="chosen")
    ax.set_xlabel(r"attribute-penalty weight $\lambda$")
    ax.set_ylabel("zero-shot AUC")
    ax.set_xticks([0.0, 0.5, 1.0, 1.5])
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}  points={list(zip(xs, ys))}")


if __name__ == "__main__":
    main()
