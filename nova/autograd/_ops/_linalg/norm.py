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
    """
    Vector or matrix norm.

    Forward: computes np.linalg.norm(input, ord=ord, axis=dim, keepdims=keepdims)
    Backward: ∂L/∂input = grad_output * input / ||input||
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        ord: int = 2,
        dim: Optional[Dim] = None,
        keepdims: bool = False,
    ) -> ndarray:
        """Compute the norm of input along given axis/dim."""
        result = np.linalg.norm(input, ord=ord, axis=dim, keepdims=keepdims)
        ctx.result = result
        ctx.ord = ord
        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for norm.

        The gradient is: grad_input = grad_output * input / ||input||
        """
        (input,) = ctx.saved_tensors
        out = ctx.result

        if ctx.ord == 2:
            if not ctx.keepdims and ctx.dim is not None:
                grad_output = np.expand_dims(grad_output, ctx.dim)
                out = np.expand_dims(out, ctx.dim)

            safe_out = np.where(out == 0, 1.0, out)
            grad_input = grad_output * (input / safe_out)
            return (grad_input,)
