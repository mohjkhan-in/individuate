"""Frozen image/text encoders (open_clip / FaRL) with an offline fallback.

`FrozenEncoders` exposes `encode_image(paths)` and `encode_text(strings)`
returning L2-normalised embeddings in a shared space. On the cluster it wraps
open_clip; with no torch / no download (or `INET_OFFLINE=1`) it falls back to a
deterministic hashing encoder so pipelines still run (embeddings are meaningless
but shaped correctly — use the synthetic regime for meaningful offline numbers).
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Sequence

import numpy as np


def _hash_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic pseudo-embedding from a string (offline fallback only)."""
    out = np.zeros(dim, dtype=np.float64)
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        rng = np.random.default_rng(h % (2 ** 32))
        out += rng.standard_normal(dim)
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


class FrozenEncoders:
    def __init__(
        self,
        backbone: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        device: str = "cpu",
        dim: int = 512,
        offline: bool | None = None,
    ):
        self.backbone = backbone
        self.pretrained = pretrained
        self.device = device
        self.dim = dim
        self.offline = (
            offline if offline is not None else os.environ.get("INET_OFFLINE") == "1"
        )
        self._model = None
        self._tokenizer = None
        self._preprocess = None
        if not self.offline:
            self._try_load()

    def _try_load(self) -> None:
        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.backbone, pretrained=self.pretrained
            )
            model.eval().to(self.device)
            self._model = model
            self._tokenizer = open_clip.get_tokenizer(self.backbone)
            self._preprocess = preprocess
            self.dim = model.text_projection.shape[1]
            self._torch = torch
        except Exception:  # noqa: BLE001 - any failure -> offline fallback
            self.offline = True

    # -- text -----------------------------------------------------------------
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if self.offline or self._model is None:
            return np.stack([_hash_vector(t, self.dim) for t in texts])
        torch = self._torch
        with torch.no_grad():
            tokens = self._tokenizer(list(texts)).to(self.device)
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    # -- image ----------------------------------------------------------------
    def encode_image(self, image_paths: Sequence[str]) -> np.ndarray:
        if self.offline or self._model is None:
            return np.stack([_hash_vector(p, self.dim) for p in image_paths])
        torch = self._torch
        from PIL import Image

        batch = torch.stack(
            [self._preprocess(Image.open(p).convert("RGB")) for p in image_paths]
        ).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()
