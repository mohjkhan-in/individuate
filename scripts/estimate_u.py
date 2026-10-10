"""Estimate the individuation scalar U(T) = I(T;Y) - I(T;A|Y) for a regime.

Offline (default): synthetic regime, numpy backend. Prints U for each caption
source so you can see individuation rise from `attribute` to `generated`.

Cluster: `--config configs/celeba.yaml` loads real data + encoders and uses the
torch backend (wire the encoders in `_encode_real`).
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import numpy as np

from individuatenet.data.synthetic import SOURCE_INDIVIDUATION, make_regime
from individuatenet.mi import estimate_u
from individuatenet.utils import load_config, set_seed


def run_synthetic(args) -> None:
    set_seed(args.seed)
    reg = make_regime(
        n_ids=args.n_ids, imgs_per_id=args.imgs_per_id,
        dim=args.dim, attr_dim=args.attr_dim, seed=args.seed,
    )
    print(f"{'source':>14} {'I(T;Y)':>8} {'I(T;A|Y)':>9} {'U':>7}")
    for source, p in SOURCE_INDIVIDUATION.items():
        text_emb = reg.caption_embeddings(p, per_image=True)
        est = estimate_u(
            text_emb, reg.image_emb, reg.id_labels, reg.attr,
            backend=args.backend, seed=args.seed,
        )
        print(f"{source:>14} {est.i_ty:8.3f} {est.i_ta_given_y:9.3f} {est.u:7.3f}")


def run_real(args) -> None:  # pragma: no cover - cluster path
    """Estimate U(T) on real embeddings for each baseline caption source.

    Loads a real dataset + frozen encoders, builds per-image captions at three
    individuation levels, and reports the bracketed U = I(T;Y) - I(T;A|Y) with
    the parametric torch backend. U should rise attribute < discriminative < rich.
    """
    from individuatenet.data.real import load_dataset
    from individuatenet.data.captions import build_captions, _ATTR_PHRASES
    from individuatenet.models.encoders import FrozenEncoders
    from individuatenet.mi.torch_estimators import TorchUConfig

    cfg = load_config(args.config)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    ds = load_dataset(cfg)
    encoders = FrozenEncoders(
        backbone=cfg.encoders.backbone,
        pretrained=cfg.encoders.pretrained,
        device=cfg.get("device", "cpu"),
        offline=cfg.encoders.get("offline", False),
    )

    # Identity-GROUPED sampling. U needs several images per identity: the torch
    # I(T;Y) anchors each caption to its identity's MEAN image embedding and
    # matches per-row, so singleton identities collapse the bound into image-
    # level retrieval (inflated, caption-insensitive) and the conditional leak
    # term has no within-identity variation to condition on. Bucket by
    # identity, keep those with >= min_imgs, take up to per_id images each.
    by_id: dict = {}
    for s in ds:
        by_id.setdefault(s.identity, []).append(s)
    ids = [i for i, lst in by_id.items() if len(lst) >= args.min_imgs]
    ids = [ids[k] for k in rng.permutation(len(ids))]
    samples: list = []
    for i in ids:
        imgs = by_id[i]
        order = rng.permutation(len(imgs))
        samples.extend(imgs[k] for k in order[: args.per_id])
        if len(samples) >= args.max_samples:
            break
    samples = samples[: args.max_samples]
    n_ids = len({s.identity for s in samples})
    print(f"[info] {len(samples)} images over {n_ids} identities "
          f"(>= {args.min_imgs} imgs/id, <= {args.per_id} sampled/id)")

    attr_keys: list = []
    for s in samples:
        cand = [k for k in _ATTR_PHRASES if k in s.attributes]
        if cand:
            attr_keys = cand
            break
    if not attr_keys:
        attr_keys = ["Male"]

    image_emb = encoders.encode_image([s.image_path for s in samples])
    attr = np.asarray(
        [[int(s.attributes.get(k, 0)) for k in attr_keys] for s in samples],
        dtype=np.float64,
    )
    id_labels = np.asarray([s.identity for s in samples])
    freq = attr.mean(axis=0)

    u_cfg = cfg.u
    backend = u_cfg.get("backend", "torch")
    torch_cfg = TorchUConfig(
        temperature=u_cfg.get("temperature", 0.07),
        critic_steps=u_cfg.get("critic_steps", 2000),
        critic_lr=u_cfg.get("critic_lr", 1e-3),
        device=cfg.get("device", "cpu"),
    )

    print(f"{'source':>14} {'I(T;Y)':>8} {'I(T;A|Y)':>9} {'U':>7}")
    for source in ["attribute", "discriminative", "rich"]:
        captions = []
        for s in samples:
            if source == "discriminative":
                active = [k for k in attr_keys if int(s.attributes.get(k, 0)) == 1]
                active.sort(key=lambda k: freq[attr_keys.index(k)])
                cap = build_captions(s.attributes, "discriminative",
                                     distinctive_keys=active[:3])
            else:
                cap = build_captions(s.attributes, source)
            captions.append(cap)
        text_emb = encoders.encode_text(captions)
        est = estimate_u(
            text_emb, image_emb, id_labels, attr,
            backend=backend, temperature=torch_cfg.temperature,
            seed=seed, torch_cfg=torch_cfg,
        )
        print(f"{source:>14} {est.i_ty:8.3f} {est.i_ta_given_y:9.3f} {est.u:7.3f}")
    if encoders.offline:
        print("\n[warning] encoders ran OFFLINE (hashing fallback): U values are "
              "pipeline-only. Use the cluster with open_clip for real numbers.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate U(T) for a regime.")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--n_ids", type=int, default=200)
    ap.add_argument("--imgs_per_id", type=int, default=20)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--attr_dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=2000,
                    help="real path: number of images to estimate U over")
    ap.add_argument("--min_imgs", type=int, default=4,
                    help="real path: keep identities with >= this many images")
    ap.add_argument("--per_id", type=int, default=10,
                    help="real path: max images sampled per identity")
    args = ap.parse_args()

    if args.config and not args.synthetic:
        run_real(args)
    else:
        run_synthetic(args)


if __name__ == "__main__":
    main()
