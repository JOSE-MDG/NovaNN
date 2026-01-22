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
    """
    Binary Cross Entropy (BCE) Loss operation.

    Computes element-wise binary cross entropy:

        L = -[target * log(input + eps) + (1 - target) * log(1 - input + eps)]

    Supports optional element-wise weighting and reduction modes.

    Forward:
        loss_i = - (y_i * log(x_i + ε) + (1 - y_i) * log(1 - x_i + ε))
        If weight is provided: loss_i = loss_i * weight_i

    Backward:
        ∂L/∂input = (sigmoid derivative) = [(1 - target)/(1 - input + eps)] - [target/(input + eps)]

    Reduction:
        - 'mean': Average over all elements
        - 'sum': Sum over all elements
        - 'none': Return element-wise loss
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        target: ndarray,
        weight: Optional[ndarray] = None,
        reduction: LossReduction = "mean",
    ) -> ndarray:
        """Compute BCE = -(target * log(input) + (1 - target) * log(1 - input))"""
        ctx.reduction = reduction
        ctx.save_for_backward(input, target, weight)
        ctx.saved_shapes = input.shape

        eps = 1e-12
        loss = -(
            target * np.log(input + eps) + (1.0 - target) * np.log(1.0 - input + eps)
        )
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
        """

        input, target, weight = ctx.saved_tensors
        input_shape = ctx.saved_shapes

        eps = 1e-12

        inv_input = 1.0 / (input + eps)
        inv_inv = 1.0 / (1 - input + eps)
        grad_input = (1 - target) * inv_inv - target * inv_input

        if weight is not None:
            grad_input *= weight

        grad_input *= grad_output

        if ctx.reduction == "mean":
            grad_input = grad_input / input.size

        return (unbroadcasting(grad_input, input_shape), None)
