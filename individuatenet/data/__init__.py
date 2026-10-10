"""Data: synthetic regimes, caption builders, and real dataset loaders."""

from .synthetic import Regime, make_regime, split_context_probe, SOURCE_INDIVIDUATION
from .captions import build_captions

__all__ = ["Regime", "make_regime", "split_context_probe", "SOURCE_INDIVIDUATION", "build_captions"]
