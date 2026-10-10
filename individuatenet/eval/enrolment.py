"""Zero-shot enrolment protocol (E1).

An unseen identity is enrolled from *text only* (one caption embedding per
identity = the gallery). Probe *images* are then matched against the text
gallery. We report verification metrics (EER, AUC + CI) over genuine/impostor
pairs and closed-set identification (Rank-k).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .metrics import delong_auc_ci, equal_error_rate, rank_k_accuracy, roc_auc


def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


@dataclass
class EnrolmentResult:
    eer: float
    auc: float
    auc_ci: tuple
    rank_k: Dict[int, float]
    n_probes: int
    n_gallery: int

    def as_row(self) -> Dict[str, float]:
        row = {"EER": self.eer, "AUC": self.auc}
        row.update({f"Rank-{k}": v for k, v in self.rank_k.items()})
        return row


def zero_shot_enrolment(
    enrol_text_emb: np.ndarray,   # (num_ids, D) one caption per identity (gallery)
    enrol_ids: Sequence[int],     # (num_ids,)
    probe_img_emb: np.ndarray,    # (num_probes, D)
    probe_ids: Sequence[int],     # (num_probes,)
    ks: Sequence[int] = (1, 5, 10),
    max_impostor_pairs: int = 2_000_000,
    seed: int = 0,
) -> EnrolmentResult:
    """Run the text-enrolment / image-probe matching protocol."""
    rng = np.random.default_rng(seed)
    gallery = _l2(np.asarray(enrol_text_emb, dtype=np.float64))
    probes = _l2(np.asarray(probe_img_emb, dtype=np.float64))
    enrol_ids = np.asarray(enrol_ids)
    probe_ids = np.asarray(probe_ids)

    sim = probes @ gallery.T  # (num_probes, num_ids) cosine similarities

    # -- verification pairs ---------------------------------------------------
    id_to_col = {int(i): c for c, i in enumerate(enrol_ids)}
    valid = np.array([int(p) in id_to_col for p in probe_ids])
    sim_v = sim[valid]
    probe_ids_v = probe_ids[valid]
    gen_cols = np.array([id_to_col[int(p)] for p in probe_ids_v])

    genuine = sim_v[np.arange(len(gen_cols)), gen_cols]

    # impostor scores: all off-diagonal, optionally subsampled
    mask = np.ones(sim_v.shape, dtype=bool)
    mask[np.arange(len(gen_cols)), gen_cols] = False
    impostor = sim_v[mask]
    if impostor.size > max_impostor_pairs:
        impostor = rng.choice(impostor, size=max_impostor_pairs, replace=False)

    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    scores = np.concatenate([genuine, impostor])

    eer = equal_error_rate(labels, scores)
    auc = roc_auc(labels, scores)
    ci = delong_auc_ci(labels, scores)

    # -- identification -------------------------------------------------------
    rk = rank_k_accuracy(sim_v, probe_ids_v, enrol_ids, ks)

    return EnrolmentResult(
        eer=eer,
        auc=auc,
        auc_ci=(ci["ci_low"], ci["ci_high"]),
        rank_k=rk,
        n_probes=int(valid.sum()),
        n_gallery=len(enrol_ids),
    )
