from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from nova.autograd._ops.utils import unbroadcasting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients, Size


@registry_op("extend")
class Extend(Function):
    """
    Broadcast an array to a new shape.

    Forward: out = broadcast_to(a, size)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, size: Size) -> ndarray:
        ctx.saved_shapes = input.shape
        return np.broadcast_to(input, shape=size)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for extend.

        The gradient is reduced back to the original shape
        by summing over broadcasted dimensions.
        """
        org_shape = ctx.saved_shapes
        grad_output = unbroadcasting(grad_output, org_shape)
        return (grad_output,)
