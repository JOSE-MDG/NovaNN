from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd._ops.utils import unbroadcasting
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("maximum")
class Maximum(Function):
    """
    Element-wise maximum of two tensors.

    Forward:
        out = max(input, other)

    Backward:
        ∂input = 1.0 where input > other, 0.5 where input == other
        ∂other = 1.0 where other > input, 0.5 where input == other
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Computes element-wise maximum."""
        ctx.save_for_backward(input, other)
        ctx.saved_shapes = (input.shape, other.shape)
        return np.maximum(input, other)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Distributes gradient between input and other, handling ties with 0.5."""
        input, other = ctx.saved_tensors
        shape_input, shape_other = ctx.saved_shapes

        mask_input = input > other
        mask_other = ~mask_input
        mask_eq = input == other

        grad_input = mask_input + 0.5 * mask_eq
        grad_other = mask_other + 0.5 * mask_eq

        return (
            unbroadcasting(grad_output * grad_input, shape_input),
            unbroadcasting(grad_output * grad_other, shape_other),
        )


@registry_op("minimum")
class Minimum(Function):
    """
    Element-wise minimum of two tensors.

    Forward:
        out = min(input, other)

    Backward:
        ∂input = 1.0 where input < other, 0.5 where input == other
        ∂other = 1.0 where other < input, 0.5 where input == other
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Computes element-wise minimum."""
        ctx.save_for_backward(input, other)
        ctx.saved_shapes = (input.shape, other.shape)
        return np.minimum(input, other)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Distributes gradient between input and other, handling ties with 0.5."""
        input, other = ctx.saved_tensors
        shape_input, shape_other = ctx.saved_shapes

        mask_input = input < other
        mask_other = ~mask_input
        mask_eq = input == other

        grad_input = mask_input + 0.5 * mask_eq
        grad_other = mask_other + 0.5 * mask_eq

        return (
            unbroadcasting(grad_output * grad_input, shape_input),
            unbroadcasting(grad_output * grad_other, shape_other),
        )


@registry_op("where")
class Where(Function):
    """
    Element-wise selection based on a condition.

    Forward:
        out = input where condition is True, else other

    Backward:
        ∂input = grad_output where condition is True
        ∂other = grad_output where condition is False
    """

    @staticmethod
    def forward(
        ctx: Context, condition: ndarray, input: ndarray, other: ndarray
    ) -> ndarray:
        """Selects elements from input or other based on condition."""
        ctx.save_for_backward(condition)
        ctx.saved_shapes = (input.shape, other.shape)
        return np.where(condition, input, other)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Routes gradients only through the selected branches."""
        (condition,) = ctx.saved_tensors
        shape_input, shape_other = ctx.saved_shapes

        grad_input = np.where(condition, grad_output, 0.0)
        grad_other = np.where(~condition, grad_output, 0.0)

        return (
            None,  # Condition is not differentiable
            unbroadcasting(grad_input, shape_input),
            unbroadcasting(grad_other, shape_other),
        )
