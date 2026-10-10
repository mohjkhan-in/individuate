"""Synthetic face regimes with planted identity/attribute structure.

This is the **offline** backend: it fabricates image / caption embeddings whose
geometry reproduces the phenomena the paper studies, so the E1/E2 harness can be
exercised end-to-end with numpy alone (no encoders, no datasets, no torch).

Generative model (per identity ``i``):
  - identity latent   ``z_i ~ N(0, I_D)``
  - attribute profile ``pi_i`` (per-identity Bernoulli means over A attributes)
  - image ``n``: draw attributes ``a_n ~ Bernoulli(pi_i)`` (within-identity
    variation = "changed looks"); embedding
        ``img = s_id * z_i + s_attr * (a_n @ W_attr) + noise``.
  - caption embedding at *individuation* ``p in [0,1]``:
        ``txt = p * s_id * z_i + (1-p) * s_attr * (a @ W_attr) + noise``.

At ``p=1`` a caption points at the identity axis (generalises across the
identity's changed looks -> small invariance gap, high U). At ``p=0`` it points
only at mutable attributes (matches just the look it describes -> large gap,
low U). Sweeping ``p`` yields the ``gap(p) = g0 - kappa*p`` law.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Maps the E1 caption sources to a synthetic individuation level p in [0, 1].
# `generated` (IndividuateNet) sits at the individuating end by construction.
SOURCE_INDIVIDUATION = {
    "attribute": 0.0,
    "discriminative": 0.4,
    "rich": 0.85,
    "generated": 1.0,
}


@dataclass
class Regime:
    name: str
    image_emb: np.ndarray      # (N, D)
    attr: np.ndarray           # (N, A) binary
    id_labels: np.ndarray      # (N,)
    # generative params retained so captions can be synthesised at any p
    _z: np.ndarray = field(repr=False, default=None)          # (num_ids, D)
    _w_attr: np.ndarray = field(repr=False, default=None)     # (A, D)
    _id_attr_mean: np.ndarray = field(repr=False, default=None)  # (num_ids, A)
    _s_id: float = 1.0
    _s_attr: float = 1.0
    _noise: float = 0.35
    _seed: int = 0

    @property
    def ids(self) -> np.ndarray:
        return np.unique(self.id_labels)

    def caption_embeddings(self, p: float, per_image: bool = True) -> np.ndarray:
        """Caption embeddings at individuation level ``p``.

        ``per_image``: one caption per image (aligned to ``image_emb`` rows, for
        U estimation). Otherwise one *enrolment* caption per identity, built from
        the identity's mean attribute profile (the "canonical" description).
        """
        # Decorrelate the caption-noise stream from the regime's generation RNG.
        # (Using the same seed would make the first draw here equal the identity
        # latent ``z`` drawn first in ``make_regime`` -- leaking identity.)
        rng = np.random.default_rng(
            (self._seed + 1) * 1_000_003 + int(round(p * 1000)) * 7 + int(per_image) * 13
        )
        if per_image:
            id_part = self._s_id * self._z[self._index_of(self.id_labels)]
            attr_part = self._s_attr * (self.attr @ self._w_attr)
            noise = self._noise * rng.standard_normal(self.image_emb.shape)
            return p * id_part + (1.0 - p) * attr_part + noise
        # per-identity enrolment caption
        id_part = self._s_id * self._z
        attr_part = self._s_attr * (self._id_attr_mean @ self._w_attr)
        noise = self._noise * rng.standard_normal(self._z.shape)
        return p * id_part + (1.0 - p) * attr_part + noise

    def _index_of(self, labels: np.ndarray) -> np.ndarray:
        order = {int(i): k for k, i in enumerate(self.ids)}
        return np.array([order[int(y)] for y in labels])


def make_regime(
    name: str = "synthetic",
    n_ids: int = 200,
    imgs_per_id: int = 20,
    dim: int = 128,
    attr_dim: int = 16,
    noise: float = 1.5,
    s_id: float = 1.0,
    s_attr: float = 1.0,
    attr_spread: float = 0.3,
    n_attr_groups: int = 1,
    seed: int = 0,
) -> Regime:
    """Build a synthetic regime. Fewer ``imgs_per_id`` -> sparse-imagery control.

    Attributes are drawn from ``n_attr_groups`` *shared* prototype profiles. The
    default ``n_attr_groups=1`` makes attributes identity-independent, so
    attribute captions sit at chance (the paper's headline finding); identity is
    carried only by the unique latent ``z``. Use ``n_attr_groups>1`` for the E4
    attribute-collision regime. ``noise`` is interpreted *relative* to the
    signal norm. The identity latent and attribute component are scaled to
    comparable norm.
    """
    rng = np.random.default_rng(seed)

    z = rng.standard_normal((n_ids, dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8

    # unit-norm attribute directions, then normalise so a typical multi-hot
    # attribute vector maps to roughly unit norm (balanced with z).
    w_attr = rng.standard_normal((attr_dim, dim))
    w_attr /= np.linalg.norm(w_attr, axis=1, keepdims=True) + 1e-8
    w_attr /= np.sqrt(max(attr_dim * 0.5, 1.0))

    # shared attribute prototypes: identities in the same group are
    # attribute-indistinguishable. K=1 -> attributes carry no identity info.
    k = max(1, min(n_attr_groups, n_ids))
    group_proto = 0.5 + attr_spread * (rng.uniform(size=(k, attr_dim)) - 0.5) * 2.0
    groups = rng.integers(0, k, size=n_ids)
    id_attr_prob = group_proto[groups]

    # per-component noise std, scaled so ``noise`` is relative to signal norm
    noise_std = noise * np.sqrt((s_id ** 2 + s_attr ** 2) / dim)

    imgs, attrs, labels = [], [], []
    for i in range(n_ids):
        a = (rng.uniform(size=(imgs_per_id, attr_dim)) < id_attr_prob[i]).astype(np.float64)
        img = s_id * z[i] + s_attr * (a @ w_attr)
        img = img + noise_std * rng.standard_normal(img.shape)
        imgs.append(img)
        attrs.append(a)
        labels.append(np.full(imgs_per_id, i))

    return Regime(
        name=name,
        image_emb=np.concatenate(imgs, axis=0),
        attr=np.concatenate(attrs, axis=0),
        id_labels=np.concatenate(labels, axis=0),
        _z=z,
        _w_attr=w_attr,
        _id_attr_mean=id_attr_prob,
        _s_id=s_id,
        _s_attr=s_attr,
        _noise=noise_std,
        _seed=seed,
    )


def split_context_probe(
    regime: Regime, test_frac: float = 0.5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-identity split into (context_idx, probe_idx).

    Context images stand in for training/enrolment material; probe images are
    held out for evaluation. Returns index arrays into the regime's rows.
    """
    rng = np.random.default_rng(seed)
    context, probe = [], []
    for i in regime.ids:
        idx = np.where(regime.id_labels == i)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * (1.0 - test_frac))))
        context.append(idx[:cut])
        probe.append(idx[cut:] if len(idx) > cut else idx[cut - 1 : cut])
    return np.concatenate(context), np.concatenate(probe)
