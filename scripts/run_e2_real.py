"""E2 (real): the training-free U predicts zero-shot enrolment.

For each caption source (attribute / discriminative / rich / generated) on a real
dataset we estimate the individuation scalar U(T)=I(T;Y)-I(T;A|Y) *without any
biometric training* (numpy backend over frozen encoders) and, on the SAME
disjoint split, measure zero-shot enrolment AUC/EER. Pooling the resulting
(U, AUC) points across sources and datasets gives the Fig. 2 scatter and the
Spearman(U, AUC) correlation of Table 2.

U is computed on context-image rows: each context image contributes its own
image + attribute vector, paired with its identity's caption embedding (probe
images are never used), so U measures caption-side individuation only.

Cluster-only (needs torch + open_clip + data). Mirrors scripts/run_e1.py's real
path for data loading and caption building.

Example:
  python scripts/run_e2_real.py --config configs/vggface2.yaml \
    --generator_ckpt $HOME/inet_ckpt/vggface2_s0/generator.pt \
    --rich_captions results/rich_vggface2.json \
    --max_ids 300 --min_imgs 6 --max_probe_per_id 4 --seed 0 \
    --out results/e2_real_vggface2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from individuatenet.eval import zero_shot_enrolment
from individuatenet.eval.predict_u import spearman, pearson
from individuatenet.mi import estimate_u
from individuatenet.utils import set_seed


def _modal_attr(imgs, keys):
    out = {}
    for k in keys:
        vals = [int(s.attributes.get(k, 0)) for s in imgs]
        out[k] = 1 if sum(vals) * 2 >= len(vals) else 0
    return out


def _encode_images(encoders, paths, bs=64):
    """Mini-batched image encode (FrozenEncoders.encode_image runs the whole list
    in one forward pass, which OOMs on thousands of context images)."""
    if not paths:
        return np.zeros((0, encoders.dim), dtype=np.float32)
    chunks = [encoders.encode_image(paths[i:i + bs]) for i in range(0, len(paths), bs)]
    return np.concatenate(chunks, axis=0)


def run(args) -> dict:
    from individuatenet.utils import load_config
    from individuatenet.data.real import load_dataset
    from individuatenet.data.captions import build_captions, _ATTR_PHRASES
    from individuatenet.models.encoders import FrozenEncoders

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    ds = load_dataset(cfg)
    encoders = FrozenEncoders(
        backbone=cfg.encoders.backbone,
        pretrained=cfg.encoders.pretrained,
        device=cfg.get("device", "cpu"),
        offline=cfg.encoders.get("offline", False),
    )

    by_id: dict = {}
    for s in ds:
        by_id.setdefault(s.identity, []).append(s)
    ids = [i for i, v in by_id.items() if len(v) >= args.min_imgs]
    rng.shuffle(ids)
    if args.max_ids:
        ids = ids[: args.max_ids]

    attr_keys: list = []
    for i in ids:
        cand = [k for k in _ATTR_PHRASES if k in by_id[i][0].attributes]
        if cand:
            attr_keys = cand
            break
    if not attr_keys:
        attr_keys = ["Male"]

    # disjoint context / probe split (probe never used for U or captions)
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
    probe_emb = _encode_images(encoders, probe_paths, bs=args.batch_size)
    probe_ids = np.asarray(probe_ids)

    # encode every context image once; build aligned per-row id / attr / image arrays
    ctx_rows_paths, ctx_rows_ids, ctx_rows_attr = [], [], []
    for i in ids:
        for s in ctx_imgs[i]:
            ctx_rows_paths.append(s.image_path)
            ctx_rows_ids.append(i)
            ctx_rows_attr.append([int(s.attributes.get(k, 0)) for k in attr_keys])
    ctx_img_emb = _encode_images(encoders, ctx_rows_paths, bs=args.batch_size)
    ctx_ids_arr = np.asarray(ctx_rows_ids)
    ctx_attr_arr = np.asarray(ctx_rows_attr, dtype=np.float64)

    enrol_attrs = {i: _modal_attr(ctx_imgs[i], attr_keys) for i in ids}
    freq = {k: float(np.mean([enrol_attrs[i][k] for i in ids])) for k in attr_keys}

    sources = ["attribute", "discriminative", "rich"]
    gen_caps = None
    if args.generator_ckpt:
        from individuatenet.models.generator import generate_captions, load_generator
        ctx_mean = np.stack([
            _encode_images(encoders, [s.image_path for s in ctx_imgs[i]],
                           bs=args.batch_size).mean(axis=0)
            for i in ids
        ])
        gmodel = load_generator(args.generator_ckpt, device=cfg.get("device", "cpu"))
        gcaps = generate_captions(gmodel, ctx_mean, device=cfg.get("device", "cpu"),
                                  max_new_tokens=cfg.generator.get("max_new_tokens", 40))
        gen_caps = dict(zip(ids, gcaps))
        sources = sources + ["generated"]

    rich_caps = None
    if args.rich_captions:
        rich_caps = json.loads(Path(args.rich_captions).read_text(encoding="utf-8"))

    id_pos = {i: p for p, i in enumerate(ids)}  # identity -> row index in per-id caption array

    points: dict = {}
    for source in sources:
        captions = []
        for i in ids:
            modal = enrol_attrs[i]
            if source == "generated":
                cap = gen_caps[i]
            elif source == "discriminative":
                active = [k for k in attr_keys if modal[k] == 1]
                active.sort(key=lambda k: freq[k])
                cap = build_captions(modal, "discriminative",
                                     distinctive_keys=active[: args.distinctive_k])
            elif source == "rich":
                free_text = None
                if rich_caps is not None:
                    for s in ctx_imgs[i]:
                        if s.image_path in rich_caps:
                            free_text = rich_caps[s.image_path]
                            break
                cap = build_captions(modal, "rich", free_text=free_text)
            else:
                cap = build_captions(modal, source)
            captions.append(cap)

        per_id_text = encoders.encode_text(captions)              # (num_ids, D)
        # expand per-identity caption embedding to each context image row
        row_text = np.stack([per_id_text[id_pos[i]] for i in ctx_rows_ids])

        est = estimate_u(row_text, ctx_img_emb, ctx_ids_arr, ctx_attr_arr,
                         backend="numpy", seed=seed)
        res = zero_shot_enrolment(
            per_id_text, ids, probe_emb, probe_ids, ks=tuple(args.ks), seed=seed
        )
        points[source] = {
            "u": est.u, "i_ty": est.i_ty, "i_ta_given_y": est.i_ta_given_y,
            "auc": res.auc, "eer": res.eer,
        }
        print(f"{source:>14}  U={est.u:+.3f}  (i_ty={est.i_ty:.3f} "
              f"i_ta|y={est.i_ta_given_y:.3f})  AUC={res.auc:.3f}  EER={100*res.eer:.2f}")

    us = [points[s]["u"] for s in points]
    aucs = [points[s]["auc"] for s in points]
    gaps = [points[s]["eer"] for s in points]
    corr = {
        "spearman_u_auc": spearman(us, aucs),
        "pearson_u_eer": pearson(us, gaps),
    }
    print(f"\n  Spearman(U, AUC) = {corr['spearman_u_auc']:+.3f}   "
          f"Pearson(U, EER) = {corr['pearson_u_eer']:+.3f}  (over {len(us)} sources)")
    if encoders.offline:
        print("\n[warning] encoders ran OFFLINE (hashing fallback): numbers are "
              "pipeline-only, not meaningful.")
    return {"dataset": cfg.data.name, "seed": seed, "points": points, "corr": corr}


def main() -> None:
    ap = argparse.ArgumentParser(description="E2 real: U predicts enrolment.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min_imgs", type=int, default=6)
    ap.add_argument("--max_ids", type=int, default=0)
    ap.add_argument("--max_probe_per_id", type=int, default=4)
    ap.add_argument("--distinctive_k", type=int, default=3)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--batch_size", type=int, default=64,
                    help="image-encode mini-batch size (lower if CUDA OOM)")
    ap.add_argument("--generator_ckpt", type=str, default=None)
    ap.add_argument("--rich_captions", type=str, default=None)
    ap.add_argument("--out", type=str, default="results/e2_real.json")
    args = ap.parse_args()

    print(f"\nE2 (real) -- U predicts enrolment ({args.config})\n")
    payload = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
