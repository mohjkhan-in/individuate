"""Generate E5 ablation configs from a base config.

Each variant toggles exactly one component of the training reward (Eq. gen) so
the ablation isolates its contribution. Writes YAMLs under
``configs/ablation/e5/`` with a per-variant ``output_dir`` (so default checkpoint
paths don't collide) and leaves every other hyper-parameter at the base value.

Variants:
  full       base config (descriptive generator; all shaping on)
  noattr     lambda_attr = 0            (remove attribute-leakage penalty)
  nofluency  beta_fluency = 0           (remove KL-to-base fluency anchor)
  confab     ground/face/propnoun/offtopic/rep weights = 0  (recovers the
             confabulating generator: no grounding, no safety shaping)
  lam1.0     lambda_attr = 1.0          (stronger attribute suppression)
  lam1.5     lambda_attr = 1.5          (over-suppression -> generic template)

Usage:
  python scripts/gen_ablation_configs.py --base configs/celeba.yaml \
    --out_dir configs/ablation/e5
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

VARIANTS = {
    "full": {},
    "noattr": {"lambda_attr": 0.0},
    "nofluency": {"beta_fluency": 0.0},
    "confab": {
        "ground_weight": 0.0, "face_weight": 0.0, "propnoun_weight": 0.0,
        "offtopic_weight": 0.0, "rep_weight": 0.0,
    },
    "lam1.0": {"lambda_attr": 1.0},
    "lam1.5": {"lambda_attr": 1.5},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/celeba.yaml")
    ap.add_argument("--out_dir", default="configs/ablation/e5")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, overrides in VARIANTS.items():
        cfg = copy.deepcopy(base)
        cfg.setdefault("generator", {})
        for k, v in overrides.items():
            cfg["generator"][k] = v
        cfg["output_dir"] = f"results/ablation/e5_{name}"
        dst = out_dir / f"{name}.yaml"
        dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        tag = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(base)"
        print(f"wrote {dst}  [{tag}]")


if __name__ == "__main__":
    main()
