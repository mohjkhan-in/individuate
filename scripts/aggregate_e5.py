"""Aggregate the E5 ablation: read per-variant E1 outputs and print the table.

Each variant was trained with one reward component toggled (see
gen_ablation_configs.py) and evaluated with scripts/run_e1.py, whose JSON holds
the ``generated`` source metrics. We tabulate zero-shot AUC/EER per variant and
emit LaTeX-ready rows for tab:ablation.

Usage:
  python scripts/aggregate_e5.py --dir results --prefix e1_e5_
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = [
    ("full", r"\method{} (full)"),
    ("noattr", r"\;$-$ attribute penalty ($\lambda{=}0$)"),
    ("nofluency", r"\;$-$ fluency (KL) term"),
    ("confab", r"\;$-$ grounding (confab.)"),
    ("lam1.0", r"\;$\lambda{=}1.0$"),
    ("lam1.5", r"\;$\lambda{=}1.5$ (over-suppress)"),
]


def _get(d, src="generated"):
    m = d.get(src, {})
    auc = m.get("AUC", [float("nan")])
    eer = m.get("EER", [float("nan")])
    auc = auc[0] if isinstance(auc, (list, tuple)) else auc
    eer = eer[0] if isinstance(eer, (list, tuple)) else eer
    return auc, eer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--prefix", default="e1_e5_")
    args = ap.parse_args()
    root = Path(args.dir)

    print(f"{'variant':>22}  {'AUC':>6}  {'EER%':>6}")
    rows = []
    for key, label in ORDER:
        f = root / f"{args.prefix}{key}.json"
        if not f.exists():
            print(f"{key:>22}  (missing {f.name})")
            rows.append((label, None, None))
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        auc, eer = _get(d)
        print(f"{key:>22}  {auc:6.3f}  {100*eer:6.2f}")
        rows.append((label, auc, eer))

    print("\n% --- LaTeX rows for tab:ablation (ZS AUC / ZS EER) ---")
    for label, auc, eer in rows:
        if auc is None:
            print(f"    {label:<34} & \\tbd & \\tbd \\\\")
        else:
            print(f"    {label:<34} & {auc:.3f} & {100*eer:.2f} \\\\")


if __name__ == "__main__":
    main()
