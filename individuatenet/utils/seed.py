"""Reproducible seeding across numpy, random and (optionally) torch."""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 0, deterministic: bool = True) -> int:
    """Seed all RNGs we might touch. Torch is seeded only if importable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
