from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils.decorators import no_inplace_op
from nova.utils import registry_op
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients

__all__ = ["GetItem", "SetItem"]


def _sanitize_index(index):
    if isinstance(index, slice):
        return index

    arr = np.asarray(index)

    if arr.dtype == np.bool_:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.int64)
    elif not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)

    return arr


@no_inplace_op
@registry_op("getitem")
class GetItem(Function):
    """
    Tensor indexing operation.

    Forward:
        out = input[index]

    Backward:
        ∂L/∂input accumulates grad_output at indexed positions
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, index) -> ndarray:
        """Extract elements from input using indexing."""
        if isinstance(index, tuple):
            actual_idx = tuple(_sanitize_index(i) for i in index)
        else:
            actual_idx = _sanitize_index(index)

        ctx.save_for_backward(input)
        ctx.idx = actual_idx

        return input[actual_idx]

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for getitem.

        The gradient is accumulated at the indexed positions.
        """
        (input,) = ctx.saved_tensors
        grad_input = np.zeros_like(input, dtype=grad_output.dtype)

        np.add.at(grad_input, ctx.idx, grad_output)

        return (grad_input,)


@no_inplace_op
@registry_op("setitem")
class SetItem(Function):
    """
    In-place item assignment.

    Forward:
        input[key] = value

    Backward:
        ∂L/∂input = grad_output with zeros at assigned positions
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
