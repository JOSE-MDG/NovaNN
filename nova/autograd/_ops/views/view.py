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
    @staticmethod
    def forward(ctx: Context, a: ndarray, size: Size) -> ndarray:
        ctx.saved_shapes = a.shape
        return np.reshape(a, size)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        org_shape = ctx.saved_shapes

        grad_output = grad_output.reshape(org_shape)

        return (grad_output, None)
