from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("relu")
class ReLU(Function):
    """
    Rectified Linear Unit activation function.

    Forward: out = max(0, x)
    Backward: ∂L/∂x = ∂L/∂out * (1 if x > 0 else 0)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Computes the forward pass of ReLU."""
        ctx.save_for_backward(input)
        return np.maximum(input, 0)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Computes the gradient of ReLU."""
        (input,) = ctx.saved_tensors
        grad_input = grad_output * (input > 0)
        return (grad_input,)
