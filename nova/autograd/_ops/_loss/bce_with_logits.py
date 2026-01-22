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

sigmoid = lambda input: 1.0 / (1.0 + np.exp(-input))  # noqa: E731


class BCEWithLogitsLoss(Function):
    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        target: ndarray,
        weight: Optional[ndarray] = None,
        reduction: LossReduction = "mean",
        pos_weight: Optional[ndarray] = None,
    ) -> ndarray:
        ctx.reduction = reduction
        ctx.save_for_backward(input, weight, pos_weight)
        ctx.saved_shapes = input.shape

        max_val = np.maximum(input, 0.0)
        log_term = np.log1p(np.exp(-np.abs(input)))

        if pos_weight is not None:
            log_weight = 1.0 + (pos_weight - 1.0) * target
            loss = max_val - input * target + log_weight * log_term
        else:
            loss = max_val - input * target + log_term

        if weight is not None:
            if weight.shape != target.shape:
                raise ValueError(
                    f"weights and targets must have the same shape, "
                    f"{weight.shape} != {target.shape}"
                )
            loss = loss * weight

        return reduce(loss, reduction)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        input, target, weight, pos_weight = ctx.saved_tensors
        input_shape = ctx.saved_shapes

        # sigmoid(input)
        sig = sigmoid(input)

        grad_input = sig - target

        if pos_weight is not None:
            grad_input *= 1.0 + (pos_weight - 1.0) * target

        if weight is not None:
            grad_input *= weight

        grad_input *= grad_output

        if ctx.reduction == "mean":
            grad_input /= input.size

        return (unbroadcasting(grad_input, input_shape), None)
