from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients, Dim


@registry_op("norm")
class Norm(Function):
    @staticmethod
    def forward(
        ctx: Context,
        a: ndarray,
        ord: int = 2,
        dim: Optional[Dim] = None,
        keepdims: bool = False,
    ) -> ndarray:

        result = np.linalg.norm(a, ord=ord, axis=dim, keepdims=keepdims)
        ctx.result = result
        ctx.ord = ord
        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.save_for_backward(a)

        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors
        out = ctx.result

        if ctx.ord == 2:

            if not ctx.keepdims and ctx.dim is not None:
                grad_output = np.expand_dims(grad_output, ctx.dim)
                out = np.expand_dims(out, ctx.dim)

            safe_out = np.where(out == 0, 1.0, out)

            grad_a = grad_output * (a / safe_out)

            grad_a = np.where(grad_a == 0, 0.0, out)

            return (grad_a,)
