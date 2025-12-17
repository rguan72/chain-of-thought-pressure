"""Reproducibility utilities."""

import os
import random
from typing import Optional

import numpy as np
import torch

_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_seed() -> Optional[int]:
    """Get the currently set global seed.

    Returns:
        The global seed if set, None otherwise.
    """
    return _GLOBAL_SEED
