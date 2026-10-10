"""Individuation estimator: U(T) = I(T;Y) - I(T;A|Y).

- `np_estimators`  training-free numpy proxies (offline, run anywhere).
- `torch_estimators` parametric InfoNCE / CLUB nets (paper-grade, cluster).
- `individuation`  the U functional dispatching over backends.
"""

from .individuation import UEstimate, estimate_u

__all__ = ["UEstimate", "estimate_u"]
