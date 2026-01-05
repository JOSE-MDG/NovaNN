from __future__ import annotations
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("setitem")
class SetItem(Function):
    """
    In-place item assignment.

    Forward:
        input[key] = value

    Backward:
        ∂input = grad_output with zeros at assigned positions
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, key: Any, value: Any) -> ndarray:
        """Assign values to input at the given index."""
        ctx.save_for_backward(key)
        input[key] = value
        return input

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for setitem.

        The gradient is zero at the assigned indices.
        """
        (key,) = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[key] = 0.0
        return (grad_input,)
