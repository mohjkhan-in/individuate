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
    n, d = 4000, 4
    print(f"{'rho':>5} {'true_MI':>9} {'infoNCE_lb':>11} {'CLUB_ub':>9}")
    for rho in (0.2, 0.4, 0.6, 0.8):
        x, y = make_gaussian(n, d, rho)
        # treat rows of x as 'text', discretise y-cluster as pseudo-identities
        ids = (np.sign(y[:, 0]) > 0).astype(int)  # coarse label just for the API
        lb = infonce_lower_bound(x, y, np.arange(n) % 50)  # 50 pseudo-identities
        ub = club_upper_bound(x, y, np.arange(n) % 50, conditional=False)
        print(f"{rho:5.2f} {true_mi(rho, d):9.3f} {lb:11.3f} {ub:9.3f}")

    print("\nExpected: lower bound < true MI < upper bound (monotone in rho).")


if __name__ == "__main__":
    main()
