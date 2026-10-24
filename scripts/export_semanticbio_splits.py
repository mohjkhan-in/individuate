#!/usr/bin/env python3
"""Export the SemanticBio (ICVGIP) open-set test-identity splits.

This reproduces, byte-for-byte, the identity hold-out used by the prior
SemanticBio paper (``src/stage1_data_pipeline.py::_partition_identities``) so the
``tab:main`` \\method{} row can be evaluated on *exactly* the same novel
identities as the published baselines.

SemanticBio protocol (from ``configs/config.yaml`` / ``configs/vggface2.yaml``):

    CelebA    : min_samples=10, holdout_identity_frac=0.20, seed=42
    VGGFace2  : min_samples=10, holdout_identity_frac=0.20, seed=42,
                max_identities=2000 (keep most-populous, ties by name)

The split itself is computed on the dataset's *string* identity keys
(``sorted(keys)`` then ``random.Random(seed).shuffle``), identical to the
original.  We then emit each novel identity as the **integer** label that
IndividuateNet's ``CelebADataset`` / ``VGGFace2Dataset`` assign
(``int(id_str)`` for CelebA, ``int(name.lstrip("n"))`` for VGGFace2), so the
file can be passed directly to ``run_e1_unified.py --split_ids``.

Run this on the cluster where the data is staged, e.g.::

    python scripts/export_semanticbio_splits.py \\
        --celeba_identity_file /ssd_scratch/mohjkhan/celeba/identity_CelebA.txt \\
        --celeba_image_dir     /ssd_scratch/mohjkhan/celeba/img_align_celeba \\
        --vggface2_root        /ssd_scratch/mohjkhan/vggface2/train \\
        --out_dir              splits

Either dataset may be omitted; only the splits whose inputs are provided are
written.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List


# --- faithful copy of stage1_data_pipeline._partition_identities (string keys) ---
def _partition_identities(identity_map: Dict[str, List[Path]],
                          holdout_frac: float, seed: int) -> List[str]:
    """Return the sorted list of *novel* (held-out) string identity keys."""
    identities = sorted(identity_map.keys())
    rng = random.Random(seed)
    rng.shuffle(identities)
    n = len(identities)
    n_novel = int(round(n * holdout_frac))
    n_novel = max(1, min(n_novel, n - 1)) if n > 1 else 0
    return sorted(identities[:n_novel])


def _collect_celeba(identity_file: Path, image_dir: Path | None,
                    min_samples: int) -> Dict[str, List[Path]]:
    """Mirror CelebAImageCollector.collect_image_paths (string identity keys)."""
    id_to_files: Dict[str, List[Path]] = {}
    with open(identity_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            fname, identity_id = parts[0], parts[1]
            if image_dir is not None:
                if not (image_dir / fname).exists():
                    continue
            id_to_files.setdefault(identity_id, []).append(image_dir / fname
                                                            if image_dir else Path(fname))
    return {k: sorted(v) for k, v in id_to_files.items() if len(v) >= min_samples}


def _collect_vggface2(img_root: Path, min_samples: int,
                      max_identities: int | None) -> Dict[str, List[Path]]:
    """Mirror VGGFace2ImageCollector.collect_image_paths (string identity keys)."""
    id_to_files: Dict[str, List[Path]] = {}
    for person_dir in sorted(img_root.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted(person_dir.glob("*.jpg"))
        if len(imgs) >= min_samples:
            id_to_files[person_dir.name] = imgs
    if max_identities is not None and len(id_to_files) > max_identities:
        ranked = sorted(id_to_files.items(),
                        key=lambda kv: (-len(kv[1]), kv[0]))[:max_identities]
        id_to_files = {k: v for k, v in ranked}
    return id_to_files


def _celeba_to_int(key: str) -> int:
    return int(key)


def _vggface2_to_int(key: str) -> int:
    stripped = key.lstrip("n")
    return int(stripped) if stripped else 0


def _write(ids_int: List[int], path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(i) for i in ids_int) + "\n", encoding="utf-8")
    print(f"[export] {label}: wrote {len(ids_int)} novel identities -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--celeba_identity_file", type=str, default=None,
                    help="path to identity_CelebA.txt")
    ap.add_argument("--celeba_image_dir", type=str, default=None,
                    help="path to img_align_celeba/ (optional; enables existence "
                         "filtering identical to the original pipeline)")
    ap.add_argument("--vggface2_root", type=str, default=None,
                    help="path to VGGFace2 train/ (folder-per-identity)")
    ap.add_argument("--min_samples", type=int, default=10)
    ap.add_argument("--holdout_frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vgg_max_identities", type=int, default=2000)
    ap.add_argument("--out_dir", type=str, default="splits")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.celeba_identity_file:
        img_dir = Path(args.celeba_image_dir) if args.celeba_image_dir else None
        id_map = _collect_celeba(Path(args.celeba_identity_file), img_dir,
                                 args.min_samples)
        print(f"[export] CelebA: {len(id_map)} identities with >= {args.min_samples} images")
        novel_keys = _partition_identities(id_map, args.holdout_frac, args.seed)
        novel_int = sorted(_celeba_to_int(k) for k in novel_keys)
        _write(novel_int, out_dir / "celeba_semanticbio_test_ids.txt", "CelebA")

    if args.vggface2_root:
        id_map = _collect_vggface2(Path(args.vggface2_root), args.min_samples,
                                   args.vgg_max_identities)
        print(f"[export] VGGFace2: {len(id_map)} identities "
              f"(min {args.min_samples}, cap {args.vgg_max_identities})")
        novel_keys = _partition_identities(id_map, args.holdout_frac, args.seed)
        novel_int = sorted(_vggface2_to_int(k) for k in novel_keys)
        _write(novel_int, out_dir / "vggface2_semanticbio_test_ids.txt", "VGGFace2")

    if not args.celeba_identity_file and not args.vggface2_root:
        ap.error("provide --celeba_identity_file and/or --vggface2_root")


if __name__ == "__main__":
    main()
