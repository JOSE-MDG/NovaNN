from __future__ import annotations
from numpy import ndarray
import numpy as np
from typing import TYPE_CHECKING
from nova.utils import registry_op
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova._typing import Gradients
    from nova.autograd.engine import Context


@registry_op("det")
class Det(Function):
    """
    Determinant of a square matrix.

    Forward: out = det(input)
    Backward: ∂L/∂input = ∂L/∂out * det(input) * (input^-T)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the determinant of a square matrix."""
        result = np.linalg.det(input)
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for determinant.

        The gradient is: grad_input = det(input) * grad_output * input^-T
        """
        input, det_val = ctx.saved_tensors
        inv_T = np.linalg.inv(input).T

        grad_input = det_val * grad_output * inv_T
        return (grad_input,)
