"""Verification / identification metrics (pure numpy).

- `roc_auc`          area under the ROC curve.
- `equal_error_rate` EER (operating point where FAR == FRR).
- `delong_auc_ci`    AUC with a DeLong variance-based confidence interval.
- `rank_k_accuracy`  closed-set Rank-k identification accuracy.
- `bootstrap_auc_ci` paired-bootstrap CI as a distribution-free alternative.

Convention: label 1 = genuine / match, 0 = impostor; higher score = more genuine.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _split(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    return scores[labels == 1], scores[labels == 0]


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """AUC via the Mann-Whitney U statistic (ties handled by mid-ranks)."""
    pos, neg = _split(np.asarray(labels), np.asarray(scores))
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(m + n, dtype=np.float64)
    ranks[order] = _midranks(np.concatenate([pos, neg])[order])
    auc = (ranks[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    return float(auc)


def _midranks(sorted_or_any: np.ndarray) -> np.ndarray:
    """Average (mid) ranks of `x`, 1-based; handles ties."""
    x = np.asarray(sorted_or_any, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def equal_error_rate(labels: Sequence[int], scores: Sequence[float]) -> float:
    """EER: the rate where false-accept (FAR) equals false-reject (FRR)."""
    pos, neg = _split(np.asarray(labels), np.asarray(scores))
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thresholds = np.unique(np.concatenate([pos, neg]))
    # FAR(t) = P(neg >= t), FRR(t) = P(pos < t)
    neg_sorted = np.sort(neg)
    pos_sorted = np.sort(pos)
    far = 1.0 - np.searchsorted(neg_sorted, thresholds, side="left") / len(neg)
    frr = np.searchsorted(pos_sorted, thresholds, side="left") / len(pos)
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0)


def _fast_delong(pos: np.ndarray, neg: np.ndarray):
    """Fast DeLong (Sun & Xu) for a single classifier: returns (auc, var)."""
    m, n = len(pos), len(neg)
    tx = _midranks(pos)
    ty = _midranks(neg)
    tz = _midranks(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    sx = np.var(v01, ddof=1) if m > 1 else 0.0
    sy = np.var(v10, ddof=1) if n > 1 else 0.0
    var = sx / m + sy / n
    return float(auc), float(var)


def delong_auc_ci(labels: Sequence[int], scores: Sequence[float], alpha: float = 0.05):
    """AUC with a DeLong (asymptotic normal) confidence interval."""
    pos, neg = _split(np.asarray(labels), np.asarray(scores))
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    auc, var = _fast_delong(pos, neg)
    se = float(np.sqrt(max(var, 0.0)))
    z = _normal_quantile(1.0 - alpha / 2.0)
    return {
        "auc": auc,
        "se": se,
        "ci_low": max(0.0, auc - z * se),
        "ci_high": min(1.0, auc + z * se),
    }


def _normal_quantile(p: float) -> float:
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except Exception:  # noqa: BLE001 - scipy optional
        # Acklam rational approximation (good to ~1e-9)
        a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
             1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
        b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
             6.680131188771972e01, -1.328068155288572e01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
             -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
             3.754408661907416e00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = np.sqrt(-2 * np.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > phigh:
            q = np.sqrt(-2 * np.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def bootstrap_auc_ci(labels, scores, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Paired-bootstrap CI on AUC (distribution-free)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if labels[idx].sum() in (0, len(idx)):
            continue
        aucs.append(roc_auc(labels[idx], scores[idx]))
    aucs = np.asarray(aucs)
    return {
        "auc": float(roc_auc(labels, scores)),
        "ci_low": float(np.quantile(aucs, alpha / 2.0)),
        "ci_high": float(np.quantile(aucs, 1.0 - alpha / 2.0)),
    }


def rank_k_accuracy(
    sim: np.ndarray,
    probe_labels: Sequence[int],
    gallery_labels: Sequence[int],
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[int, float]:
    """Closed-set Rank-k: fraction of probes whose true identity appears in the
    top-k gallery entries by similarity. `sim` is (num_probes, num_gallery)."""
    sim = np.asarray(sim)
    probe_labels = np.asarray(probe_labels)
    gallery_labels = np.asarray(gallery_labels)
    order = np.argsort(-sim, axis=1)  # descending similarity
    ranked_labels = gallery_labels[order]
    out: Dict[int, float] = {}
    for k in ks:
        topk = ranked_labels[:, :k]
        hit = (topk == probe_labels[:, None]).any(axis=1)
        out[int(k)] = float(hit.mean())
    return out
