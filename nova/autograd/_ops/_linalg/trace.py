from __future__ import annotations
from numpy import ndarray
import numpy as np
from typing import TYPE_CHECKING
from nova.utils import registry_op
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova._typing import Gradients
    from nova.autograd.engine import Context


@registry_op("trace")
class Trace(Function):
    """
    Trace of a square matrix.

    Forward: out = sum(diag(input))
    Backward: ∂L/∂input = grad_output * I
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the trace (sum of diagonal elements) of a square matrix."""
        ctx.save_for_backward(input)
        return np.linalg.trace(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for trace.

        The gradient is: grad_input = grad_output * identity matrix
        """
        (input,) = ctx.saved_tensors
        N = input.shape[0]
        grad_input = np.eye(N) * grad_output
        return (grad_input,)
