from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from .utils import reduce
from numpy import ndarray
from nova.autograd.function import Function
from nova.autograd._ops.utils import unbroadcasting

if TYPE_CHECKING:
    from nova._typing import Gradients, LossReduction
    from nova.autograd.engine import Context


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
        ctx.save_for_backward(input, target, weight)
        ctx.saved_shapes = (input.shape, target.shape)

        loss = (input - target) ** 2

        if weight is not None:
            if weight.shape != target.shape:
                raise ValueError(
                    f"weights and targets must be have the same shape, {weight.shape} != {target.shape}"
                )
            loss = loss * weight

        return reduce(loss, reduction)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for mse loss.

        Gradient:
            ∂L/∂input  =  2 * (input - target) * grad_output
        """
        input, target, weight = ctx.saved_tensors
        input_shape = ctx.saved_shapes

        grad_input = 2.0 * (input - target)

        if weight is not None:
            grad_input *= weight

        if ctx.reduction == "mean":
            grad_input /= input.size

        grad_input *= grad_output
        return (unbroadcasting(grad_input, input_shape), None)
