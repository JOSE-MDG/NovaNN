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
    @staticmethod
    def forward(ctx: Context, a: ndarray, size: Size) -> ndarray:
        ctx.saved_shapes = a.shape
        return np.broadcast_to(a, shape=size)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        org_shape = ctx.saved_shapes

        grad_output = unbroadcasting(grad_output, org_shape)

        return (grad_output,)
