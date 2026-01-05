from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING
from nova.utils import registry_op
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova._typing import Gradients
    from nova.autograd.engine import Context


@registry_op("inv")
class Inv(Function):
    """
    Matrix inversion.

    Forward: out = inv(input)
    Backward: ∂L/∂input = -inv(input)^T · grad_output · inv(input)^T
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse of a square matrix."""
        result = np.linalg.inv(input)
        ctx.save_for_backward(result.T)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for inverse matrix.

        The gradient is: grad_input = -inv(input)^T · grad_output · inv(input)^T
        """
        (inv_T,) = ctx.saved_tensors
        grad_input = -inv_T @ grad_output @ inv_T
        return (grad_input,)
