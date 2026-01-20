from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.autograd._ops.utils import unbroadcasting

if TYPE_CHECKING:
    from nova._typing import Gradients, LossReduction
    from nova.autograd.engine import Context


def _reduce(
    loss: ndarray,
    reduction_mode: LossReduction = "mean",
    batch_size: Optional[int] = None,
) -> ndarray:
    """
    Applies reduction to loss array based on specified mode.

    Args:
        loss: Unreduced loss array.
        reduction_mode: Type of reduction to apply.
            - 'none': No reduction, returns full loss tensor
            - 'sum': Sums all elements
            - 'mean': Averages all elements

    Returns:
        Reduced loss tensor.

    Raises:
        ValueError: If reduction_mode is invalid or batch_size is missing
            for 'batchmean' mode.
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


class MSELoss(Function):
    """
    Mean Squared Error (MSE) / L2 Loss operation.

    Computes the squared difference between each element in the input and target.
    This implementation is an 'atomic' operation, which is numerically more
    stable than computing (a - b) ** 2 through separate subtraction and
    power operations.

    Forward:
        L = (input - target)^2
        If weight is provided: L = weight * (input - target)^2

    Backward:
        ∂L/∂input  =  2 * (input - target) * grad_output
        ∂L/∂target = -2 * (input - target) * grad_output = 2 * (target - input) * grad_output

    Reduction:
        - 'mean': Gradients are scaled by 1/N (N = total number of elements).
        - 'sum':  Gradients are not scaled.
        - 'none': Gradients are returned element-wise.
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        target: ndarray,
        reduction: LossReduction = "mean",
        weight: Optional[ndarray] = None,
    ) -> ndarray:
        """
        Computes (input - target) ** 2.
        """
        ctx.reduction = reduction
        # Save weight even if it's None to maintain the tuple structure in backward
        ctx.save_for_backward(input, target, weight)
        ctx.saved_shapes = (input.shape, target.shape)

        # Core calculation: (y_pred - y_true)^2
        loss = (input - target) ** 2

        if weight is not None:
            loss = loss * weight

        return _reduce(loss, reduction_mode=reduction)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for mse loss.

        Gradient:
            ∂L/∂input  =  2 * (input - target) * grad_output
            ∂L/∂target = -2 * (input - target) * grad_output = 2 * (target - input) * grad_output
        """
        input, target, weight = ctx.saved_tensors
        input_shape, target_shape = ctx.saved_shapes

        grad_input = 2.0 * (input - target)

        if weight is not None:
            grad_input *= weight

        if ctx.reduction == "mean":
            grad_input /= input.size

        grad_input *= grad_output
        grad_target = -grad_input

        return (
            unbroadcasting(grad_input, input_shape),
            unbroadcasting(grad_target, target_shape),
        )
