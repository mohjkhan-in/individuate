"""E4 (real): individuation is not attribute richness (attribute-collision test).

Standard enrolment lets any other identity act as an impostor. Here we constrain
impostors to the probe identity's attribute cohort: identities whose modal
MAAD/CelebA attribute profile is within Hamming distance ``tau`` of the probe's
(i.e. they share the described attributes). A caption that individuates only by
naming shared attributes collapses toward chance in this setting, because every
cohort member carries the same attribute words; a caption that encodes genuinely
identity-specific facial structure retains its margin.

For each caption source we report both the unconstrained AUC/EER and the
collision-restricted AUC/EER, plus the drop. Expectation: attribute and
discriminative sources drop toward 0.5, while the IndividuateNet generator holds.

Cluster-only (needs torch + open_clip + data). Mirrors run_e2_real.py loading.

Example:
  python scripts/run_e4_real.py --config configs/vggface2.yaml \
    --generator_ckpt $HOME/inet_ckpt/vggface2_s0/generator.pt \
    --rich_captions results/rich_vggface2.json \
    --max_ids 300 --min_imgs 6 --max_probe_per_id 4 --tau 2 --seed 0 \
    --out results/e4_real_vggface2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from individuatenet.eval.metrics import equal_error_rate, roc_auc


def _modal_attr(imgs, keys):
    out = {}
    for k in keys:
        vals = [int(s.attributes.get(k, 0)) for s in imgs]
        out[k] = 1 if sum(vals) * 2 >= len(vals) else 0
    return out


def _encode_images(encoders, paths, bs=64):
    if not paths:
        return np.zeros((0, encoders.dim), dtype=np.float32)
    chunks = [encoders.encode_image(paths[i:i + bs]) for i in range(0, len(paths), bs)]
    return np.concatenate(chunks, axis=0)


def _l2(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _auc_eer(sim, probe_ids, enrol_ids, cohort_mask=None):
    """Pool genuine vs impostor cosine scores and return (auc, eer, n_imp).

    cohort_mask[a, b] = True iff gallery identity b is an admissible impostor for
    gallery identity a. When None, every off-diagonal identity is an impostor.
    """
    id_to_col = {int(i): c for c, i in enumerate(enrol_ids)}
    genuine, impostor = [], []
    for r, pid in enumerate(probe_ids):
        pid = int(pid)
        if pid not in id_to_col:
            continue
        gy = id_to_col[pid]
        genuine.append(sim[r, gy])
        if cohort_mask is None:
            cols = [c for c in range(sim.shape[1]) if c != gy]
        else:
            cols = np.where(cohort_mask[gy])[0]
        if len(cols):
            impostor.extend(sim[r, cols].tolist())
    if not genuine or not impostor:
        return float("nan"), float("nan"), len(impostor)
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    scores = np.concatenate([np.asarray(genuine), np.asarray(impostor)])
    return roc_auc(labels, scores), equal_error_rate(labels, scores), len(impostor)


def run(args) -> dict:
    from individuatenet.utils import load_config, set_seed
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

    # disjoint context / probe split
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
    probe_emb = _l2(_encode_images(encoders, probe_paths, bs=args.batch_size))
    probe_ids = np.asarray(probe_ids)

    enrol_attrs = {i: _modal_attr(ctx_imgs[i], attr_keys) for i in ids}
    freq = {k: float(np.mean([enrol_attrs[i][k] for i in ids])) for k in attr_keys}

    # attribute-cohort mask over the gallery via Hamming distance of modal profiles
    profiles = np.array([[enrol_attrs[i][k] for k in attr_keys] for i in ids], dtype=int)
    ham = (profiles[:, None, :] != profiles[None, :, :]).sum(axis=-1)
    cohort_mask = (ham <= args.tau)
    np.fill_diagonal(cohort_mask, False)
    cohort_sizes = cohort_mask.sum(axis=1)
    print(f"attr cohort (Hamming<= {args.tau} over {len(attr_keys)} attrs): "
          f"mean {cohort_sizes.mean():.1f}, median {np.median(cohort_sizes):.0f}, "
          f"min {cohort_sizes.min()}, max {cohort_sizes.max()}")

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

        gallery = _l2(encoders.encode_text(captions))
        sim = probe_emb @ gallery.T

        auc_all, eer_all, n_all = _auc_eer(sim, probe_ids, ids, cohort_mask=None)
        auc_col, eer_col, n_col = _auc_eer(sim, probe_ids, ids, cohort_mask=cohort_mask)
        points[source] = {
            "auc": auc_all, "eer": eer_all,
            "auc_collision": auc_col, "eer_collision": eer_col,
            "auc_drop": auc_all - auc_col,
            "n_impostor": n_all, "n_impostor_collision": n_col,
        }
        print(f"{source:>14}  AUC {auc_all:.3f} -> collision {auc_col:.3f} "
              f"(drop {auc_all - auc_col:+.3f})   EER {100*eer_all:.2f} -> "
              f"{100*eer_col:.2f}")

    if encoders.offline:
        print("\n[warning] encoders ran OFFLINE (hashing fallback): numbers are "
              "pipeline-only, not meaningful.")
    return {
        "dataset": cfg.data.name, "seed": seed, "tau": args.tau,
        "n_attr_keys": len(attr_keys),
        "cohort_mean": float(cohort_sizes.mean()),
        "points": points,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="E4 real: attribute-collision test.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min_imgs", type=int, default=6)
    ap.add_argument("--max_ids", type=int, default=0)
    ap.add_argument("--max_probe_per_id", type=int, default=4)
    ap.add_argument("--distinctive_k", type=int, default=3)
    ap.add_argument("--tau", type=int, default=2,
                    help="max Hamming distance of modal attribute profiles to be "
                         "counted as an attribute-cohort impostor")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--generator_ckpt", type=str, default=None)
    ap.add_argument("--rich_captions", type=str, default=None)
    ap.add_argument("--out", type=str, default="results/e4_real.json")
    args = ap.parse_args()

    print(f"\nE4 (real) -- attribute-collision ({args.config})\n")
    payload = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
