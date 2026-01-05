from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients, Size


@registry_op("view")
class View(Function):
    """
    Reshape an array without copying data.

    Forward: out = reshape(a, size)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, size: Size) -> ndarray:
        ctx.saved_shapes = input.shape
        return np.reshape(input, size)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for view.

        The gradient is reshaped back to the original input shape.
        """
        org_shape = ctx.saved_shapes
        return (grad_output.reshape(org_shape),)
