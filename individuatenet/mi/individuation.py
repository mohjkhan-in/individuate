"""The individuation functional U(T) = I(T;Y) - I(T;A|Y).

Dispatches over the offline numpy proxy and the parametric torch backend, and
returns a bracketed estimate (lower bound on I(T;Y), upper bound on the leak)
so U is a conservative lower bound on true individuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UEstimate:
    u: float                 # U = I(T;Y) - I(T;A|Y)
    i_ty: float              # lower bound on I(T;Y)
    i_ta_given_y: float      # upper bound on I(T;A|Y)
    backend: str

    def as_dict(self) -> dict:
        return {
            "u": self.u,
            "i_ty": self.i_ty,
            "i_ta_given_y": self.i_ta_given_y,
            "backend": self.backend,
        }


def estimate_u(
    text_emb: np.ndarray,
    image_emb: np.ndarray,
    id_labels: np.ndarray,
    attr: np.ndarray,
    backend: str = "numpy",
    temperature: float = 0.07,
    ridge_lambda: float = 1e-2,
    seed: int = 0,
    torch_cfg=None,
) -> UEstimate:
    """Estimate U(T). `backend` is 'numpy' (offline) or 'torch' (parametric)."""
    id_labels = np.asarray(id_labels)
    attr = np.asarray(attr, dtype=np.float64)

    if backend == "numpy":
        from .np_estimators import estimate_u_numpy

        est = estimate_u_numpy(
            text_emb, image_emb, id_labels, attr,
            temperature=temperature, ridge_lambda=ridge_lambda, seed=seed,
        )
        return UEstimate(est.u, est.i_ty, est.i_ta_given_y, "numpy")

    if backend == "torch":
        from .torch_estimators import TorchUConfig, estimate_i_ta, estimate_i_ty

        cfg = torch_cfg or TorchUConfig(temperature=temperature)
        # paired identity anchor: mean image embedding of each sample's identity
        anchor = np.zeros_like(text_emb)
        for i in np.unique(id_labels):
            mask = id_labels == i
            anchor[mask] = image_emb[mask].mean(axis=0)
        # conditioning signal for the conditional leak term
        cond = anchor
        i_ty = estimate_i_ty(text_emb, anchor, cfg)
        i_ta = estimate_i_ta(text_emb, attr, cond=cond, cfg=cfg)
        return UEstimate(i_ty - i_ta, i_ty, i_ta, "torch")

    raise ValueError(f"unknown backend {backend!r} (use 'numpy' or 'torch')")
