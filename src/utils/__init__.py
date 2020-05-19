import torch
import numpy as np
import random

from .visualize import sentinel_as_tci, plot_roc_curve, plot_with_mask
from .regions import STR2BB


__all__ = ["sentinel_as_tci", "set_seed", "STR2BB", "plot_roc_curve", "plot_with_mask"]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
