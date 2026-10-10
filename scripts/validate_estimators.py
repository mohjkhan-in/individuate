"""Validate the MI estimators on planted Gaussians with known mutual information.

For correlated Gaussians X, Y (dim d, per-dim correlation rho) the true MI is
    I(X;Y) = -0.5 * d * log(1 - rho^2)   (nats).

We check that the InfoNCE-style lower bound sits below and the CLUB-style upper
bound sits above the truth (numpy backend). This is the sanity harness cited as
"validate on synthetic corpora with known MI" in the proposal.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (path setup)

import numpy as np

from individuatenet.mi.np_estimators import club_upper_bound, infonce_lower_bound


def true_mi(rho: float, d: int) -> float:
    return -0.5 * d * np.log(1 - rho ** 2)


def make_gaussian(n: int, d: int, rho: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    noise = rng.standard_normal((n, d))
    y = rho * x + np.sqrt(1 - rho ** 2) * noise
    return x, y


def main() -> None:
    # Moderate dim so the fixed cosine InfoNCE critic has enough signal; the
    # closed-form continuous MI grows with d.
    n, d = 4000, 16
    print(f"{'rho':>5} {'true_MI':>9} {'infoNCE_lb':>11} {'CLUB_ub':>9} {'ok':>4}")
    for rho in (0.2, 0.4, 0.6, 0.8):
        x, y = make_gaussian(n, d, rho)
        # InfoNCE: give every sample its own identity so the estimator reduces
        # to the standard CPC lower bound on I(X;Y) over N-1 negatives (ceiling
        # log N). temperature=0.5 suits these *untrained* synthetic embeddings;
        # the production default (0.07) targets trained CLIP features.
        lb = infonce_lower_bound(x, y, np.arange(n), temperature=0.5)
        # CLUB upper bound on the same I(X;Y) (unconditional; labels unused).
        ub = club_upper_bound(x, y, np.arange(n) % 50, conditional=False)
        truth = true_mi(rho, d)
        ok = "ok" if lb < truth < ub else "BAD"
        print(f"{rho:5.2f} {truth:9.3f} {lb:11.3f} {ub:9.3f} {ok:>4}")

    print("\nExpected: lower bound < true MI < upper bound (monotone in rho).")


if __name__ == "__main__":
    main()
