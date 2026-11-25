#!/usr/bin/env python3
"""Audit CelebA generated captions and rich-caption coverage.

Checks:
  1. whether rich_celeba.json actually matches CelebA context images;
  2. lexical safety/descriptiveness statistics;
  3. visual grounding to context and held-out probe images;
  4. whether randomized identity/caption assignments destroy the E1 gain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from individuatenet.data.real import load_dataset
from individuatenet.eval.enrolment import zero_shot_enrolment
from individuatenet.models.encoders import FrozenEncoders
from individuatenet.models.generator import generate_captions, load_generator
from individuatenet.train.train_generator import (
    count_face_tokens,
    count_offtopic_tokens,
    count_propnoun_tokens,
    count_repetition,
)
from individuatenet.utils import load_config


def rich_coverage(samples, rich_path: str | None) -> dict:
    if not rich_path:
        return {"enabled": False}

    path = Path(rich_path)
    if not path.exists():
        return {"enabled": True, "exists": False, "path": str(path)}

    rich = json.loads(path.read_text(encoding="utf-8"))
    exact_keys = {str(k) for k in rich}
    basename_keys = {Path(str(k)).name for k in rich}
    dataset_paths = [str(s.image_path) for s in samples]

    exact = sum(p in exact_keys for p in dataset_paths)
    basename = sum(Path(p).name in basename_keys for p in dataset_paths)
    unique_basenames = len({Path(p).name for p in dataset_paths})

    return {
        "enabled": True,
        "exists": True,
        "path": str(path),
        "entries": len(rich),
        "dataset_images": len(dataset_paths),
        "unique_dataset_basenames": unique_basenames,
        "exact_matches": exact,
        "basename_matches": basename,
        "exact_match_fraction": exact / max(len(dataset_paths), 1),
        "basename_match_fraction": basename / max(len(dataset_paths), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit generated CelebA captions")
    ap.add_argument("--config", default="configs/celeba.yaml")
    ap.add_argument("--generator_ckpt", required=True)
    ap.add_argument("--rich_captions", default="results/rich_celeba.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_imgs", type=int, default=4)
    ap.add_argument("--max_ids", type=int, default=300)
    ap.add_argument("--max_probe_per_id", type=int, default=10)
    ap.add_argument("--permutation_seeds", type=int, default=3)
    ap.add_argument("--out", default="results/celeba_caption_audit.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg.get("device", "cpu")
    rng = np.random.default_rng(args.seed)

    samples = load_dataset(cfg)
    rich = rich_coverage(samples, args.rich_captions)

    by_id: dict[int, list] = {}
    for sample in samples:
        by_id.setdefault(int(sample.identity), []).append(sample)

    ids = [identity for identity, images in by_id.items()
           if len(images) >= args.min_imgs]
    rng.shuffle(ids)
    if args.max_ids:
        ids = ids[:args.max_ids]

    encoders = FrozenEncoders(
        backbone=cfg.encoders.backbone,
        pretrained=cfg.encoders.pretrained,
        device=device,
        offline=cfg.encoders.get("offline", False),
    )
    if encoders.offline:
        raise RuntimeError(
            "Offline hashing fallback is active; real caption audit requires "
            "open_clip and encoders.offline=false"
        )

    context_images: dict[int, list] = {}
    probe_paths: list[str] = []
    probe_ids: list[int] = []

    rng = np.random.default_rng(args.seed)
    for identity in ids:
        images = list(by_id[identity])
        rng.shuffle(images)
        n_probe = min(args.max_probe_per_id, max(1, len(images) // 2))
        context = images[n_probe:] or images
        context_images[identity] = context
        for sample in images[:n_probe]:
            probe_paths.append(str(sample.image_path))
            probe_ids.append(identity)

    context_emb = np.stack([
        encoders.encode_image([
            str(sample.image_path) for sample in context_images[identity]
        ]).mean(axis=0)
        for identity in ids
    ])

    model = load_generator(args.generator_ckpt, device=device)
    captions = generate_captions(
        model,
        context_emb,
        device=device,
        max_new_tokens=cfg.generator.get("max_new_tokens", 40),
    )

    caption_emb = encoders.encode_text(captions)
    probe_emb = encoders.encode_image(probe_paths)
    probe_ids_arr = np.asarray(probe_ids, dtype=np.int64)

    propnoun = np.asarray(count_propnoun_tokens(captions))
    repetition = np.asarray(count_repetition(captions))
    offtopic = np.asarray(count_offtopic_tokens(captions))
    face = np.asarray(count_face_tokens(captions))

    normalized_caption = caption_emb / (
        np.linalg.norm(caption_emb, axis=1, keepdims=True) + 1e-12
    )
    nearest = normalized_caption @ normalized_caption.T
    np.fill_diagonal(nearest, -1.0)

    own_probe_cos = []
    for identity, text_embedding in zip(ids, caption_emb):
        mask = probe_ids_arr == identity
        if mask.any():
            probe_vec = probe_emb[mask]
            probe_vec = probe_vec / (
                np.linalg.norm(probe_vec, axis=1, keepdims=True) + 1e-12
            )
            text_vec = text_embedding / (
                np.linalg.norm(text_embedding) + 1e-12
            )
            own_probe_cos.append(float(np.max(probe_vec @ text_vec)))

    normal = np.asarray(ids, dtype=np.int64)
    enrolment = zero_shot_enrolment(
        caption_emb,
        normal,
        probe_emb,
        probe_ids_arr,
        ks=(1, 5, 10),
        seed=args.seed,
    )

    permutation_results = []
    for permutation_seed in range(args.permutation_seeds):
        perm_rng = np.random.default_rng(permutation_seed)
        permutation = perm_rng.permutation(len(ids))
        perm_emb = caption_emb[permutation]
        perm_result = zero_shot_enrolment(
            perm_emb,
            normal,
            probe_emb,
            probe_ids_arr,
            ks=(1, 5, 10),
            seed=permutation_seed,
        )
        permutation_results.append({
            "seed": permutation_seed,
            "EER": perm_result.eer,
            "AUC": perm_result.auc,
            "Rank-1": perm_result.rank_k[1],
        })

    unique_captions = len(set(c.strip().lower() for c in captions))
    record = {
        "seed": args.seed,
        "identity_count": len(ids),
        "probe_count": len(probe_paths),
        "rich_coverage": rich,
        "captions": dict(zip(normal.tolist(), captions)),
        "metrics": {
            "unique_captions": unique_captions,
            "unique_caption_fraction": unique_captions / max(len(captions), 1),
            "propnoun_mean": float(propnoun.mean()),
            "propnoun_rows_gt_zero": int((propnoun > 0).sum()),
            "offtopic_mean": float(offtopic.mean()),
            "offtopic_rows_gt_zero": int((offtopic > 0).sum()),
            "repetition_mean": float(repetition.mean()),
            "repetition_max": float(repetition.max()),
            "face_token_mean": float(face.mean()),
            "face_token_rows_gt_zero": int((face > 0).sum()),
            "heldout_probe_caption_cosine_mean": float(np.mean(own_probe_cos)),
            "heldout_probe_caption_cosine_max": float(np.max(own_probe_cos)),
            "nearest_other_caption_cosine_mean": float(nearest.max(axis=1).mean()),
        },
        "enrolment": {
            "EER": enrolment.eer,
            "AUC": enrolment.auc,
            "Rank-1": enrolment.rank_k[1],
        },
        "randomized_caption_control": permutation_results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(json.dumps({
        "rich_coverage": rich,
        "metrics": record["metrics"],
        "enrolment": record["enrolment"],
        "randomized_caption_control": permutation_results,
        "saved": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
