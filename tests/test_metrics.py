"""Sanity checks for the numpy metrics (no torch required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from individuatenet.eval.metrics import (
    bootstrap_auc_ci,
    delong_auc_ci,
    equal_error_rate,
    rank_k_accuracy,
    roc_auc,
)


def test_auc_perfect_separation():
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert abs(roc_auc(labels, scores) - 1.0) < 1e-9


def test_auc_random_is_half():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=4000)
    scores = rng.standard_normal(4000)
    assert abs(roc_auc(labels, scores) - 0.5) < 0.05


def test_eer_perfect_is_zero():
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert equal_error_rate(labels, scores) < 1e-6


def test_delong_ci_brackets_auc():
    rng = np.random.default_rng(1)
    labels = np.concatenate([np.ones(200), np.zeros(200)])
    scores = np.concatenate([rng.normal(1.0, 1, 200), rng.normal(0.0, 1, 200)])
    out = delong_auc_ci(labels, scores)
    assert out["ci_low"] <= out["auc"] <= out["ci_high"]


def test_bootstrap_ci_runs():
    rng = np.random.default_rng(2)
    labels = np.concatenate([np.ones(100), np.zeros(100)])
    scores = np.concatenate([rng.normal(1.0, 1, 100), rng.normal(0.0, 1, 100)])
    out = bootstrap_auc_ci(labels, scores, n_boot=200)
    assert 0.0 <= out["ci_low"] <= out["ci_high"] <= 1.0


def test_rank_k_monotone():
    sim = np.array([[0.9, 0.1, 0.2], [0.2, 0.8, 0.1]])
    probe = np.array([0, 1])
    gallery = np.array([0, 1, 2])
    rk = rank_k_accuracy(sim, probe, gallery, ks=(1, 2, 3))
    assert rk[1] <= rk[2] <= rk[3]
    assert abs(rk[1] - 1.0) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all metric tests passed")
