from __future__ import annotations
import numpy as np
import math
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("gelu")
class GELU(Function):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Uses the tanh approximation:
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Computes the forward pass using the tanh approximation."""
        inner = math.sqrt(2.0 / math.pi) * (input + 0.044715 * np.power(input, 3))
        tanh_inner = np.tanh(inner)
        out = 0.5 * input * (1.0 + tanh_inner)
        ctx.save_for_backward(input, inner)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Computes the gradient via the chain rule on the approximation."""
        input, inner = ctx.saved_tensors
        tanh_inner = np.tanh(inner)

        left = 0.5 * (1.0 + tanh_inner)
        right = (
            0.5
            * input
            * (1.0 - np.power(tanh_inner, 2))
            * math.sqrt(2.0 / math.pi)
            * (1.0 + 3.0 * 0.044715 * np.power(input, 2))
        )
        grad_input = grad_output * (left + right)
        return (grad_input,)
