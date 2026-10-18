"""E1: zero-shot enrolment main table.

For each caption source (attribute / discriminative / rich / generated) we
enrol every unseen identity from a single caption embedding and match held-out
probe images against that text gallery. Reports EER / AUC / Rank-1 (mean +/- std
over seeds) and writes a markdown + JSON table.

Offline (`--synthetic`, default): numpy only, no encoders, no data. The caption
sources map to synthetic individuation levels, so you should see EER fall and
AUC rise from `attribute` to `generated` -- the shape of Table 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from individuatenet.data.synthetic import SOURCE_INDIVIDUATION, make_regime, split_context_probe
from individuatenet.eval import zero_shot_enrolment
from individuatenet.utils import set_seed


def _agg(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def run_synthetic(args) -> dict:
    sources = list(SOURCE_INDIVIDUATION.keys())
    results = {s: {"EER": [], "AUC": [], "Rank-1": []} for s in sources}

    for seed in range(args.seeds):
        set_seed(seed)
        reg = make_regime(
            n_ids=args.n_ids, imgs_per_id=args.imgs_per_id,
            dim=args.dim, attr_dim=args.attr_dim, seed=seed,
        )
        _, probe_idx = split_context_probe(reg, test_frac=args.test_frac, seed=seed)
        probe_emb = reg.image_emb[probe_idx]
        probe_ids = reg.id_labels[probe_idx]

        for source in sources:
            p = SOURCE_INDIVIDUATION[source]
            enrol_text = reg.caption_embeddings(p, per_image=False)  # (num_ids, D)
            res = zero_shot_enrolment(
                enrol_text, reg.ids, probe_emb, probe_ids, ks=(1, 5, 10), seed=seed
            )
            results[source]["EER"].append(res.eer)
            results[source]["AUC"].append(res.auc)
            results[source]["Rank-1"].append(res.rank_k[1])

    table = {}
    for source in sources:
        table[source] = {
            metric: _agg(vals) for metric, vals in results[source].items()
        }
    return table


def _modal_attr(imgs, keys):
    """Majority-vote attribute vector for one identity (dict form)."""
    out = {}
    for k in keys:
        vals = [int(s.attributes.get(k, 0)) for s in imgs]
        out[k] = 1 if sum(vals) * 2 >= len(vals) else 0
    return out


def run_real(args) -> dict:
    """E1 on real data: text-only enrolment vs image probes, per caption source.

    Uses the frozen encoders (open_clip on the cluster; deterministic hashing
    fallback offline). The learned ``generated`` source needs a trained
    IndividuateNet checkpoint and is added once training (items 5-6) is done.
    """
    from individuatenet.utils import load_config
    from individuatenet.data.real import load_dataset
    from individuatenet.data.captions import build_captions, _ATTR_PHRASES
    from individuatenet.models.encoders import FrozenEncoders

    cfg = load_config(args.config)
    seed = args.seed if getattr(args, "seed", None) is not None else cfg.get("seed", 0)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    ds = load_dataset(cfg)
    encoders = FrozenEncoders(
        backbone=cfg.encoders.backbone,
        pretrained=cfg.encoders.pretrained,
        device=cfg.get("device", "cpu"),
        offline=cfg.encoders.get("offline", False),
    )

    # group by identity; keep image-abundant identities, take an open-set subset
    by_id: dict = {}
    for s in ds:
        by_id.setdefault(s.identity, []).append(s)
    ids = [i for i, v in by_id.items() if len(v) >= args.min_imgs]
    rng.shuffle(ids)
    if args.max_ids:
        ids = ids[: args.max_ids]

    # resolve the attribute vocabulary actually present on this dataset
    attr_keys: list = []
    for i in ids:
        cand = [k for k in _ATTR_PHRASES if k in by_id[i][0].attributes]
        if cand:
            attr_keys = cand
            break
    if not attr_keys:
        attr_keys = ["Male"]

    # disjoint context / probe split per identity: probe images are held out and
    # never used to build the generator context or the enrolment caption, so AUC
    # measures grounding to *unseen* images of the identity, not memorisation.
    ctx_imgs: dict = {}
    probe_paths, probe_ids = [], []
    for i in ids:
        imgs = list(by_id[i])
        rng.shuffle(imgs)
        n_probe = min(args.max_probe_per_id, max(1, len(imgs) // 2))
        ctx_imgs[i] = imgs[n_probe:] or imgs  # fallback for tiny identities
        for s in imgs[:n_probe]:
            probe_paths.append(s.image_path)
            probe_ids.append(i)
    probe_emb = encoders.encode_image(probe_paths)
    probe_ids = np.asarray(probe_ids)

    # per-identity modal attributes (context images only) + attribute frequency
    enrol_attrs = {i: _modal_attr(ctx_imgs[i], attr_keys) for i in ids}
    freq = {k: float(np.mean([enrol_attrs[i][k] for i in ids])) for k in attr_keys}

    sources = ["attribute", "discriminative", "rich"]
    gen_caps = None
    if getattr(args, "generator_ckpt", None):
        from individuatenet.models.generator import generate_captions, load_generator
        ctx = np.stack([
            encoders.encode_image([s.image_path for s in ctx_imgs[i]]).mean(axis=0)
            for i in ids
        ])
        gmodel = load_generator(args.generator_ckpt, device=cfg.get("device", "cpu"))
        gcaps = generate_captions(gmodel, ctx, device=cfg.get("device", "cpu"),
                                  max_new_tokens=cfg.generator.get("max_new_tokens", 40))
        gen_caps = dict(zip(ids, gcaps))
        sources = sources + ["generated"]
        if getattr(args, "dump_captions", 0):
            from individuatenet.train.train_generator import (
                count_offtopic_tokens, count_repetition,
            )
            print("\n--- generated captions (safety audit) ---")
            for i, cap in list(zip(ids, gcaps))[: args.dump_captions]:
                off = count_offtopic_tokens([cap])[0]
                rep = count_repetition([cap])[0]
                flag = "  <-- OFF-TOPIC" if off > 0 else ""
                print(f"[{i}] off={off:.0f} rep={rep:.2f} | {cap}{flag}")
            print("--- end caption dump ---\n")

    table: dict = {}
    rich_caps = None
    if getattr(args, "rich_captions", None):
        rich_caps = json.loads(Path(args.rich_captions).read_text(encoding="utf-8"))
        n_hit = sum(
            any(s.image_path in rich_caps for s in ctx_imgs[i]) for i in ids
        )
        print(f"[rich] free-text VLM captions: {n_hit}/{len(ids)} identities covered "
              f"(rest fall back to attribute text)")
    for source in sources:
        captions = []
        for i in ids:
            modal = enrol_attrs[i]
            if source == "generated":
                cap = gen_caps[i]
            elif source == "discriminative":
                active = [k for k in attr_keys if modal[k] == 1]
                active.sort(key=lambda k: freq[k])  # rarest active attrs first
                cap = build_captions(modal, "discriminative",
                                     distinctive_keys=active[: args.distinctive_k])
            elif source == "rich":
                free_text = None
                if rich_caps is not None:
                    for s in ctx_imgs[i]:  # context images only -> no probe leakage
                        if s.image_path in rich_caps:
                            free_text = rich_caps[s.image_path]
                            break
                cap = build_captions(modal, "rich", free_text=free_text)
            else:
                cap = build_captions(modal, source)
            captions.append(cap)
        gallery = encoders.encode_text(captions)  # (num_ids, D) text gallery
        res = zero_shot_enrolment(
            gallery, ids, probe_emb, probe_ids, ks=tuple(args.ks), seed=seed
        )
        table[source] = {
            "EER": (res.eer, 0.0),
            "AUC": (res.auc, 0.0),
            "Rank-1": (res.rank_k[1], 0.0),
        }
        print(f"{source:>14}  EER={100*res.eer:6.2f}  AUC={res.auc:.3f}  "
              f"Rank-1={100*res.rank_k[1]:6.2f}  "
              f"(probes={res.n_probes}, gallery={res.n_gallery})")
    if encoders.offline:
        print("\n[warning] encoders ran OFFLINE (hashing fallback): numbers are "
              "pipeline-only, not meaningful. Set encoders.offline=false with "
              "open_clip available on the cluster for real metrics.")
    return table


def _print_markdown(table: dict) -> str:
    lines = ["| Caption source | EER (%) | AUC | Rank-1 (%) |",
             "|---|---|---|---|"]
    for source, metrics in table.items():
        eer_m, eer_s = metrics["EER"]
        auc_m, auc_s = metrics["AUC"]
        r1_m, r1_s = metrics["Rank-1"]
        lines.append(
            f"| {source} | {100*eer_m:.2f} +/- {100*eer_s:.2f} | "
            f"{auc_m:.3f} +/- {auc_s:.3f} | {100*r1_m:.2f} +/- {100*r1_s:.2f} |"
        )
    md = "\n".join(lines)
    print(md)
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 zero-shot enrolment main table.")
    ap.add_argument("--synthetic", action="store_true", default=True)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_ids", type=int, default=200)
    ap.add_argument("--imgs_per_id", type=int, default=20)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--attr_dim", type=int, default=16)
    ap.add_argument("--test_frac", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="results/e1_synthetic.json")
    # real (--config) knobs
    ap.add_argument("--min_imgs", type=int, default=4,
                    help="keep identities with at least this many images")
    ap.add_argument("--max_ids", type=int, default=0,
                    help="cap number of unseen identities (0 = all)")
    ap.add_argument("--max_probe_per_id", type=int, default=10)
    ap.add_argument("--distinctive_k", type=int, default=3)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--generator_ckpt", type=str, default=None,
                    help="real path: add a 'generated' source from a trained checkpoint")
    ap.add_argument("--dump_captions", type=int, default=0,
                    help="print the first N generated captions (safety audit) before scoring")
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg.seed (real path): varies the ctx/probe split per run")
    ap.add_argument("--rich_captions", type=str, default=None,
                    help="real path: JSON {image_path: caption} from scripts/caption_vlm.py; "
                         "makes the 'rich' source a genuine free-text VLM baseline")
    args = ap.parse_args()

    if args.config:
        print(f"\nE1 -- zero-shot enrolment (real: {args.config})\n")
        table = run_real(args)
        args.out = args.out.replace("synthetic", "real")
    else:
        table = run_synthetic(args)
        print("\nE1 -- zero-shot enrolment (synthetic regime)\n")
    _print_markdown(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
