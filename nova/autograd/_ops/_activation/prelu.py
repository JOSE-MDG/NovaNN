from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("prelu")
class PReLU(Function):
    """
    Parametric Rectified Linear Unit activation function.

    Forward: out = max(0, x) + weight * min(0, x)
    Backward: Gradient is computed for both input and the learnable weight.
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, weight: float | ndarray) -> ndarray:
        """
        Computes the forward pass of PReLU.

        Args:
            input: Input data.
            weight: Learnable parameter (scalar or array).
        """
        weight_arr = np.asarray(weight)
        ctx.save_for_backward(input, weight_arr)
        return np.where(input > 0, input, weight_arr * input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Computes gradients for input and weight."""
        input, weight = ctx.saved_tensors

        grad_input = grad_output * np.where(input > 0, 1.0, weight)
        grad_weight = np.sum(grad_output * np.where(input > 0, 0.0, input))

        return (grad_input, grad_weight)
