"""E1 (unified cross-method protocol): zero-shot enrolment on the shared
open-set splits reused from SemanticBio, for the ``\\method{} (ours)`` row of
the main table (tab:main).

Difference from ``run_e1.py --config`` (our own disjoint-probe protocol): here
the gallery is the *full identity-disjoint open-set test split* (not an
image-abundant 300-id subset), so the numbers are directly comparable to the
SemanticBio baseline anchors (CLIP / MetaCLIP / FaRL / VLM-prompted). Reports
EER / AUC / Rank-1 as mean +/- std over seeds.

SemanticBio protocol (from the ICVGIP'26 paper): CelebA = 1,674 identity-disjoint
open-set test identities, 1:1 verification (text enrolment vs probe image),
EER / ROC-AUC / Rank-{1,5,10}, mean+/-std over the three seeds {7, 42, 123}; the
CLIP / MetaCLIP / FaRL / VLM-prompted anchors are scored on the identical
probe+gallery split. VGGFace2 = grounded caption-swap over 3 seeds.

For exact parity with the published anchors, pin the test-identity list with
``--split_ids ids.txt`` (one integer identity per line) using the split from the
SemanticBio repo (github.com/mohjkhan-in/semantic_biometric). Without it, a
deterministic identity-disjoint split (``data.test_frac``, varied per seed) is
used as an open-set proxy -- comparable in protocol but not identity-for-identity.

Cluster-only (needs open_clip + a trained generator checkpoint). Example::

    python scripts/run_e1_unified.py \\
        --config configs/celeba.yaml \\
        --generator_ckpt $HOME/inet_ckpt/e5_full/generator.pt \\
        --seed_list 7 42 123 --min_imgs 6 --max_probe_per_id 4 \\
        --split_ids splits/celeba_semanticbio_test_ids.txt \\
        --rich_captions results/rich_celeba.json \\
        --out results/e1_unified_celeba.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from individuatenet.data.captions import build_captions, _ATTR_PHRASES
from individuatenet.data.real import load_dataset
from individuatenet.eval import zero_shot_enrolment
from individuatenet.models.encoders import FrozenEncoders
from individuatenet.utils import load_config, set_seed


def _agg(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _modal_attr(imgs, keys):
    """Majority-vote attribute vector for one identity."""
    out = {}
    for k in keys:
        vals = [int(s.attributes.get(k, 0)) for s in imgs]
        out[k] = 1 if sum(vals) * 2 >= len(vals) else 0
    return out


def _open_set_test_ids(by_id, all_ids, test_frac, min_imgs, split_ids, rng):
    """Identity-disjoint open-set test split.

    If ``split_ids`` is given, use exactly those identities (intersected with
    the ones that have >= ``min_imgs`` images here) for parity with a pinned
    SemanticBio split. Otherwise take the held-out side of a deterministic
    identity shuffle.
    """
    if split_ids is not None:
        test = [i for i in split_ids if i in by_id and len(by_id[i]) >= min_imgs]
        return test
    shuffled = list(all_ids)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - test_frac))
    test = [i for i in shuffled[cut:] if len(by_id[i]) >= min_imgs]
    return test


def _eval_one_seed(cfg, encoders, args, seed) -> dict:
    set_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    ds = load_dataset(cfg)
    by_id: dict = {}
    for s in ds:
        by_id.setdefault(s.identity, []).append(s)
    all_ids = sorted(by_id.keys())

    split_ids = None
    if getattr(args, "split_ids", None):
        split_ids = [
            int(x) for x in Path(args.split_ids).read_text().split() if x.strip()
        ]

    test_frac = cfg.data.get("test_frac", 0.5)
    ids = _open_set_test_ids(by_id, all_ids, test_frac, args.min_imgs, split_ids, rng)
    rng.shuffle(ids)
    if args.max_ids:
        ids = ids[: args.max_ids]
    if not ids:
        raise RuntimeError("no open-set test identities after filtering "
                           f"(min_imgs={args.min_imgs}); loosen --min_imgs")

    # attribute vocabulary actually present on this dataset
    attr_keys: list = []
    for i in ids:
        cand = [k for k in _ATTR_PHRASES if k in by_id[i][0].attributes]
        if cand:
            attr_keys = cand
            break
    if not attr_keys:
        attr_keys = ["Male"]

    # disjoint context / probe split per identity (held-out probes)
    ctx_imgs: dict = {}
    probe_paths, probe_ids = [], []
    for i in ids:
        imgs = list(by_id[i])
        rng.shuffle(imgs)
        n_probe = min(args.max_probe_per_id, max(1, len(imgs) // 2))
        ctx_imgs[i] = imgs[n_probe:] or imgs
        for s in imgs[:n_probe]:
            probe_paths.append(s.image_path)
            probe_ids.append(i)
    probe_emb = encoders.encode_image(probe_paths)
    probe_ids = np.asarray(probe_ids)

    enrol_attrs = {i: _modal_attr(ctx_imgs[i], attr_keys) for i in ids}
    freq = {k: float(np.mean([enrol_attrs[i][k] for i in ids])) for k in attr_keys}

    # caption sources
    sources = ["attribute"]
    rich_caps = None
    if getattr(args, "rich_captions", None):
        rich_caps = json.loads(Path(args.rich_captions).read_text(encoding="utf-8"))
        sources.append("rich")

    gen_caps = None
    if getattr(args, "generator_ckpt", None):
        from individuatenet.models.generator import generate_captions, load_generator
        ctx = np.stack([
            encoders.encode_image([s.image_path for s in ctx_imgs[i]]).mean(axis=0)
            for i in ids
        ])
        gmodel = load_generator(args.generator_ckpt, device=cfg.get("device", "cpu"))
        gcaps = generate_captions(
            gmodel, ctx, device=cfg.get("device", "cpu"),
            max_new_tokens=cfg.generator.get("max_new_tokens", 40),
        )
        gen_caps = dict(zip(ids, gcaps))
        sources.append("generated")

    out: dict = {}
    for source in sources:
        captions = []
        for i in ids:
            modal = enrol_attrs[i]
            if source == "generated":
                cap = gen_caps[i]
            elif source == "rich":
                free_text = None
                for s in ctx_imgs[i]:
                    if s.image_path in rich_caps:
                        free_text = rich_caps[s.image_path]
                        break
                cap = build_captions(modal, "rich", free_text=free_text)
            else:
                cap = build_captions(modal, source)
            captions.append(cap)
        gallery = encoders.encode_text(captions)
        res = zero_shot_enrolment(
            gallery, ids, probe_emb, probe_ids, ks=tuple(args.ks), seed=seed
        )
        out[source] = {"EER": res.eer, "AUC": res.auc, "Rank-1": res.rank_k[1]}
        print(f"  seed {seed} {source:>12}  EER={100*res.eer:6.2f}  "
              f"AUC={res.auc:.3f}  Rank-1={100*res.rank_k[1]:6.2f}  "
              f"(probes={res.n_probes}, gallery={res.n_gallery})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="E1 unified open-set enrolment (tab:main \\method{} row).")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--seed_list", type=int, nargs="+", default=[7, 42, 123],
                    help="SemanticBio evaluation seeds (default {7,42,123})")
    ap.add_argument("--generator_ckpt", type=str, default=None,
                    help="trained IndividuateNet checkpoint -> 'generated' source")
    ap.add_argument("--rich_captions", type=str, default=None,
                    help="JSON {image_path: caption} free-text VLM baseline")
    ap.add_argument("--split_ids", type=str, default=None,
                    help="file of SemanticBio test identities (one int/line) for exact parity")
    ap.add_argument("--min_imgs", type=int, default=6,
                    help="keep test identities with at least this many images")
    ap.add_argument("--max_ids", type=int, default=0,
                    help="cap number of open-set test identities (0 = full split)")
    ap.add_argument("--max_probe_per_id", type=int, default=4)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--out", type=str, default="results/e1_unified.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    encoders = FrozenEncoders(
        backbone=cfg.encoders.backbone,
        pretrained=cfg.encoders.pretrained,
        device=cfg.get("device", "cpu"),
        offline=cfg.encoders.get("offline", False),
    )

    print(f"\nE1 unified open-set enrolment ({args.config}, seeds {args.seed_list})\n")
    per_seed = [_eval_one_seed(cfg, encoders, args, s) for s in args.seed_list]

    sources = list(per_seed[0].keys())
    table: dict = {}
    for source in sources:
        table[source] = {
            m: _agg([ps[source][m] for ps in per_seed])
            for m in ("EER", "AUC", "Rank-1")
        }

    print("\n| Enrolment text | EER (%) | AUC | Rank-1 (%) |")
    print("|---|---|---|---|")
    for source, metrics in table.items():
        (em, es), (am, asd), (rm, rs) = (
            metrics["EER"], metrics["AUC"], metrics["Rank-1"])
        print(f"| {source} | {100*em:.2f} +/- {100*es:.2f} | "
              f"{am:.3f} +/- {asd:.3f} | {100*rm:.2f} +/- {100*rs:.2f} |")

    if encoders.offline:
        print("\n[warning] encoders ran OFFLINE (hashing fallback): numbers are "
              "pipeline-only, not meaningful. Use open_clip on the cluster.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
