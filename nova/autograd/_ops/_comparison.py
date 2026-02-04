from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd._ops.utils import unbroadcasting, dispatch_output
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients

__all__ = ["Maximum", "Minimum", "Where", "Sign"]


@registry_op("maximum")
class Maximum(Function):
    """
    Element-wise maximum of two tensors.

    Forward:
        out = max(input, other)

    Backward:
        ∂L/∂input = 1 where input > other, 0.5 where equal
        ∂L/∂other = 1 where other > input, 0.5 where equal
    """

    @staticmethod
    def forward(
        ctx: Context, input: ndarray, other: ndarray, _out: Optional[ndarray] = None
    ) -> ndarray:
        """Compute element-wise maximum."""
        ctx.save_for_backward(input, other)
        ctx.saved_shapes = (input.shape, other.shape)
        return np.maximum(input, other, out=_out)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for maximum.

        The gradient is split between inputs, sharing equally in case of ties.
        """
        input, other = ctx.saved_tensors
        shape_input, shape_other = ctx.saved_shapes

        mask_input = input > other
        mask_other = other > input
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
        ∂L/∂input = 1 where input < other, 0.5 where equal
        ∂L/∂other = 1 where other < input, 0.5 where equal
    """

    @staticmethod
    def forward(
        ctx: Context, input: ndarray, other: ndarray, _out: Optional[ndarray] = None
    ) -> ndarray:
        """Compute element-wise minimum."""
        ctx.save_for_backward(input, other)
        ctx.saved_shapes = (input.shape, other.shape)
        return np.minimum(input, other, out=_out)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for minimum.

        The gradient is split between inputs, sharing equally in case of ties.
        """
        input, other = ctx.saved_tensors
        shape_input, shape_other = ctx.saved_shapes

        mask_input = input < other
        mask_other = other < input
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
    Conditional element-wise selection.

    Forward:
        out = input if condition else other

    Backward:
        ∂L/∂input = grad_output where condition is True
        ∂L/∂other = grad_output where condition is False
    """

    @staticmethod
    def forward(
        ctx: Context,
        condition: ndarray,
        input: ndarray,
        other: ndarray,
        _out: Optional[ndarray] = None,
    ) -> ndarray:
        """Select elements based on condition."""
        ctx.save_for_backward(condition)
        ctx.saved_shapes = (input.shape, other.shape)
        return dispatch_output(_out, np.where(condition, input, other))

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for where.

        The gradient flows only through the selected branch.
        """
        (condition,) = ctx.saved_tensors
        shape_input, shape_other = ctx.saved_shapes

        grad_input = np.where(condition, grad_output, 0.0)
        grad_other = np.where(~condition, grad_output, 0.0)

        return (
            None,
            unbroadcasting(grad_input, shape_input),
            unbroadcasting(grad_other, shape_other),
        )


@registry_op("sign")
class Sign(Function):
    """
    Element-wise sign function.

    Forward: out = sign(a) = {-1 if a<0, 0 if a=0, 1 if a>0}
    Backward: Not differentiable (returns None)
    """

    @staticmethod
    def forward(
        ctx: Context, input: ndarray, _out: Optional[ndarray] = None
    ) -> ndarray:
        """Computes sign(a)."""
        return np.sign(input, out=_out)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Sign function is not differentiable."""
        return (None,)
