"""Evaluation harness: metrics, enrolment protocol (E1), U-predictor (E2)."""

from .metrics import roc_auc, equal_error_rate, delong_auc_ci, rank_k_accuracy
from .enrolment import zero_shot_enrolment, EnrolmentResult
from .predict_u import fit_u_predictor, UPredictorFit

__all__ = [
    "roc_auc",
    "equal_error_rate",
    "delong_auc_ci",
    "rank_k_accuracy",
    "zero_shot_enrolment",
    "EnrolmentResult",
    "fit_u_predictor",
    "UPredictorFit",
]
