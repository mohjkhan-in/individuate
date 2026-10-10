"""E2: the training-free U-predictor.

Fits ``AUC = sigma(alpha + beta * U + gamma * log n_img)`` across regimes and
reports how well the (training-free) individuation scalar ``U`` forecasts
zero-shot enrolment success. We lead with rank correlation (Spearman), as the
paper argues, and also report the fitted coefficients and R^2 on the logit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


@dataclass
class UPredictorFit:
    alpha: float
    beta: float          # coefficient on U
    gamma: float         # coefficient on log n_img
    r2_logit: float
    spearman_u_auc: float
    pearson_u_gap: float

    def predict(self, u: float, n_img: int) -> float:
        return float(_sigmoid(self.alpha + self.beta * u + self.gamma * np.log(n_img)))


def fit_u_predictor(
    u_values: Sequence[float],
    auc_values: Sequence[float],
    n_img: Sequence[int],
    gap_values: Sequence[float] | None = None,
) -> UPredictorFit:
    """Least-squares fit on the logit of AUC plus rank/linear correlations."""
    u = np.asarray(u_values, dtype=np.float64)
    auc = np.asarray(auc_values, dtype=np.float64)
    logn = np.log(np.asarray(n_img, dtype=np.float64))
    y = _logit(auc)

    X = np.stack([np.ones_like(u), u, logn], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    pear_gap = pearson(u, gap_values) if gap_values is not None else float("nan")

    return UPredictorFit(
        alpha=float(coef[0]),
        beta=float(coef[1]),
        gamma=float(coef[2]),
        r2_logit=float(r2),
        spearman_u_auc=spearman(u, auc),
        pearson_u_gap=pear_gap,
    )
