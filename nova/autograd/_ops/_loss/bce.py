from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from numpy import ndarray
import numpy as np
from .utils import reduce
from nova.autograd._ops.utils import unbroadcasting
from nova.autograd.function import Function


if TYPE_CHECKING:
    from nova._typing import Gradients, LossReduction
    from nova.autograd.engine import Context


class BCELoss(Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        target: ndarray,
        reduction: LossReduction = "mean",
        weight: Optional[ndarray] = None,
    ) -> ndarray:
        """Compute BCE = -(target * log(input) + (1 - target) * log(1 - input))"""
        ctx.reduction = reduction
        ctx.save_for_backward(input, target, weight)
        ctx.saved_shapes = (input.shape, target.shape)

        eps = 1e-12
        loss = -(target * np.log(input + eps) + (1 - target) * np.log(1 - input))

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
        Backward pass for Binary Cross Entropy (BCE) loss.

        Gradient:
            ∂L/∂input  = [(1 - target) / (1 - input + ε)]  -  [target / (input + ε)]
            ∂L/∂target = -log(input + ε) + log(1 - input + ε)
        """

        input, target, weight = ctx.saved_tensors
        input_shape, target_shape = ctx.saved_shapes

        eps = 1e-12

        inv_input = 1.0 / (input + eps)
        inv_inv = 1.0 / (1 - input + eps)
        grad_input = (1 - target) * inv_inv - target * inv_input

        if weight is not None:
            grad_input *= weight

        grad_input *= grad_output

        if ctx.reduction == "mean":
            grad_input = grad_input / input.size

        grad_target = -np.log(input + eps) + np.log(1 - input + eps)
        grad_target *= grad_output

        if ctx.reduction == "mean":
            grad_target /= input.size

        return (
            unbroadcasting(grad_input, input_shape),
            unbroadcasting(grad_target, target_shape),
        )
