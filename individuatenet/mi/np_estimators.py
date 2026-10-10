"""Training-free numpy MI estimators for the individuation functional.

These are the *offline* backend: closed-form / non-parametric bounds that need
no gradient training, so `U` can be computed anywhere (no torch, no GPU).

- `infonce_lower_bound`  lower bound on I(T;Y) with a fixed cosine critic.
- `club_upper_bound`     CLUB upper bound on the (conditional) attribute leak.
- `estimate_u_numpy`     U(T) = I(T;Y) - I(T;A|Y).

All quantities are in **nats**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _l2_normalise(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _identity_prototypes(
    anchor_emb: np.ndarray, id_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean anchor embedding per identity. Returns (prototypes, id_order)."""
    ids = np.unique(id_labels)
    protos = np.stack([anchor_emb[id_labels == i].mean(axis=0) for i in ids])
    return protos, ids


def infonce_lower_bound(
    text_emb: np.ndarray,
    anchor_emb: np.ndarray,
    id_labels: np.ndarray,
    temperature: float = 0.07,
) -> float:
    """InfoNCE lower bound on I(T;Y) using a non-parametric cosine critic.

    The critic scores a caption against per-identity prototypes built from the
    paired modality (`anchor_emb`, e.g. mean image embedding). With `J`
    identities the bound equals ``log J - crossentropy`` in nats.
    """
    text = _l2_normalise(np.asarray(text_emb, dtype=np.float64))
    protos, ids = _identity_prototypes(np.asarray(anchor_emb, dtype=np.float64), id_labels)
    protos = _l2_normalise(protos)

    # logits: (N, J) cosine similarities / temperature
    logits = (text @ protos.T) / temperature
    # target column for each sample
    id_to_col = {int(i): c for c, i in enumerate(ids)}
    targets = np.array([id_to_col[int(y)] for y in id_labels])

    # log-softmax over identities
    logits -= logits.max(axis=1, keepdims=True)
    log_denom = np.log(np.exp(logits).sum(axis=1))
    log_prob = logits[np.arange(len(targets)), targets] - log_denom
    # E[log p(y|t)] + log J  == InfoNCE lower bound on I(T;Y)
    return float(log_prob.mean() + np.log(len(ids)))


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge weights mapping x -> y (with bias column)."""
    n, d = x.shape
    xb = np.concatenate([x, np.ones((n, 1))], axis=1)
    reg = lam * np.eye(d + 1)
    reg[-1, -1] = 0.0  # do not penalise bias
    w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    return w


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return xb @ w


def _residualise_attributes(attr: np.ndarray, id_labels: np.ndarray) -> np.ndarray:
    """Remove per-identity attribute means so the leak term is conditional on Y."""
    resid = attr.astype(np.float64).copy()
    for i in np.unique(id_labels):
        mask = id_labels == i
        resid[mask] -= resid[mask].mean(axis=0, keepdims=True)
    return resid


def club_upper_bound(
    text_emb: np.ndarray,
    attr: np.ndarray,
    id_labels: np.ndarray,
    ridge_lambda: float = 1e-2,
    conditional: bool = True,
    rng: np.random.Generator | None = None,
) -> float:
    """CLUB upper bound on I(T;A) (or I(T;A|Y) when `conditional`).

    A Gaussian variational predictor q(a|t)=N(Wt, sigma^2 I) is fit by ridge.
    CLUB = E_p(t,a)[log q(a|t)] - E_p(t)p(a)[log q(a|t)]; with a Gaussian q this
    reduces to ``(shuffled_mse - matched_mse) / (2 sigma^2)`` per dimension,
    summed over attribute dimensions. Conditioning on Y is done by first
    residualising attributes by their per-identity mean.
    """
    rng = rng or np.random.default_rng(0)
    text = np.asarray(text_emb, dtype=np.float64)
    a = np.asarray(attr, dtype=np.float64)
    if conditional:
        a = _residualise_attributes(a, id_labels)

    w = _ridge_fit(text, a, ridge_lambda)
    pred = _ridge_predict(text, w)

    matched = ((a - pred) ** 2).mean(axis=0)               # per-dim matched MSE
    # Use the marginal attribute variance as the Gaussian scale: this keeps the
    # per-dim CLUB bounded (~<=1 nat) instead of exploding when matched -> 0.
    sigma2 = np.maximum(a.var(axis=0), 1e-6)

    perm = rng.permutation(len(a))
    shuffled = ((a[perm] - pred) ** 2).mean(axis=0)

    per_dim = (shuffled - matched) / (2.0 * sigma2)
    return float(np.clip(per_dim, 0.0, None).sum())


@dataclass
class UEstimateNumpy:
    i_ty: float           # I(T;Y) lower bound
    i_ta_given_y: float   # I(T;A|Y) upper bound
    u: float              # U = I(T;Y) - I(T;A|Y)


def estimate_u_numpy(
    text_emb: np.ndarray,
    image_emb: np.ndarray,
    id_labels: np.ndarray,
    attr: np.ndarray,
    temperature: float = 0.07,
    ridge_lambda: float = 1e-2,
    seed: int = 0,
) -> UEstimateNumpy:
    """Training-free U(T) = I(T;Y) - I(T;A|Y).

    `image_emb` supplies the paired anchor for the I(T;Y) lower bound; `attr`
    is the mutable-attribute matrix (binary or continuous).
    """
    rng = np.random.default_rng(seed)
    i_ty = infonce_lower_bound(text_emb, image_emb, id_labels, temperature)
    i_ta = club_upper_bound(
        text_emb, attr, id_labels, ridge_lambda, conditional=True, rng=rng
    )
    return UEstimateNumpy(i_ty=i_ty, i_ta_given_y=i_ta, u=i_ty - i_ta)
