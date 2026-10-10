"""Real dataset loaders (CelebA / VGGFace2). Cluster-side; guarded imports.

These read the on-disk layout described in the configs and yield samples of the
form ``{"image_path", "identity", "attributes"}``. Encoding to embeddings is
done by ``models.encoders``. Kept torch-free at import time so the offline path
never pays for these.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List


@dataclass
class Sample:
    image_path: str
    identity: int
    attributes: Dict[str, int]


class CelebADataset:
    """CelebA aligned crops + list_attr / identity annotation files."""

    def __init__(self, root: str, attr_file: str, identity_file: str, image_dir: str):
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.attr_path = self.root / attr_file
        self.identity_path = self.root / identity_file
        if not self.attr_path.exists():
            raise FileNotFoundError(f"missing attribute file: {self.attr_path}")
        self._attr_names, self._attrs = self._read_attrs(self.attr_path)
        self._identities = self._read_identities(self.identity_path)

    @staticmethod
    def _read_attrs(path: Path):
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        names = lines[1].split()
        rows: Dict[str, List[int]] = {}
        for line in lines[2:]:
            parts = line.split()
            fname, vals = parts[0], parts[1:]
            rows[fname] = [1 if v == "1" else 0 for v in vals]
        return names, rows

    @staticmethod
    def _read_identities(path: Path) -> Dict[str, int]:
        ids: Dict[str, int] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                fname, ident = line.split()
                ids[fname] = int(ident)
        return ids

    def __iter__(self) -> Iterator[Sample]:
        for fname, vals in self._attrs.items():
            if fname not in self._identities:
                continue
            attrs = dict(zip(self._attr_names, vals))
            yield Sample(
                image_path=str(self.image_dir / fname),
                identity=self._identities[fname],
                attributes=attrs,
            )

    def identity_disjoint_split(self, test_frac: float = 0.5, seed: int = 0):
        """Return (train_ids, test_ids) with disjoint identities (open-set)."""
        import random

        ids = sorted({s.identity for s in self})
        rng = random.Random(seed)
        rng.shuffle(ids)
        cut = int(len(ids) * (1.0 - test_frac))
        return set(ids[:cut]), set(ids[cut:])


class VGGFace2Dataset:
    """VGGFace2 folder-per-identity + MAAD-Face attribute CSV (optional)."""

    def __init__(self, root: str, image_dir: str = "train", attr_csv: str | None = None):
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.attr_csv = self.root / attr_csv if attr_csv else None
        if not self.image_dir.exists():
            raise FileNotFoundError(f"missing image dir: {self.image_dir}")
        self._attrs = self._read_maad(self.attr_csv) if self.attr_csv else {}

    @staticmethod
    def _read_maad(path: Path) -> Dict[str, Dict[str, int]]:
        import csv

        table: Dict[str, Dict[str, int]] = {}
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            key = reader.fieldnames[0]
            attr_cols = reader.fieldnames[1:]
            for row in reader:
                table[row[key]] = {c: 1 if row[c] in ("1", "1.0") else 0 for c in attr_cols}
        return table

    def __iter__(self) -> Iterator[Sample]:
        for ident_dir in sorted(self.image_dir.iterdir()):
            if not ident_dir.is_dir():
                continue
            try:
                identity = int(ident_dir.name.lstrip("n") or "0")
            except ValueError:
                identity = abs(hash(ident_dir.name)) % (10 ** 8)
            for img in sorted(ident_dir.glob("*.jpg")):
                rel = f"{ident_dir.name}/{img.name}"
                yield Sample(
                    image_path=str(img),
                    identity=identity,
                    attributes=self._attrs.get(rel, {}),
                )


def load_dataset(cfg) -> object:
    """Factory dispatching on ``cfg.data.name``."""
    name = cfg.data.name
    root = cfg.data.get("root", "data_local")
    if name == "celeba":
        return CelebADataset(
            root,
            cfg.data.get("attr_file", "list_attr_celeba.txt"),
            cfg.data.get("identity_file", "identity_CelebA.txt"),
            cfg.data.get("image_dir", "img_align_celeba"),
        )
    if name == "vggface2":
        return VGGFace2Dataset(
            root,
            cfg.data.get("image_dir", "train"),
            cfg.data.get("attr_csv"),
        )
    raise ValueError(f"real loader for {name!r} not available (use synthetic offline)")
