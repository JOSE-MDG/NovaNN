from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import LossReduction


def reduce(
    loss: Tensor,
    reduction_mode: LossReduction = "mean",
) -> Tensor:
    """
    Applies reduction to loss tensor based on specified mode.

    Args:
        loss: Unreduced loss tensor.
        reduction_mode: Type of reduction to apply.
            - 'none': No reduction, returns full loss tensor
            - 'sum': Sums all elements
            - 'mean': Averages all elements
    Returns:
        Reduced loss tensor.

    Raises:
        ValueError: If reduction_mode is invalid.
    """
    if reduction_mode == "none":
        return loss
    elif reduction_mode == "sum":
        return np.sum(loss)
    elif reduction_mode == "mean":
        return np.mean(loss)
    else:
        raise ValueError(
            f"reduction expect ('sum','mean','none'), got '{reduction_mode}'"
        )
