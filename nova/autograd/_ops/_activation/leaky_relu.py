from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("leaky_relu")
class LeakyReLU(Function):
    """
    Leaky Rectified Linear Unit activation function.

    Forward: out = x if x > 0 else alpha * x
    Backward: ∂L/∂x = ∂L/∂out * (1 if x > 0 else alpha)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, alpha: float = 0.01) -> ndarray:
        """
        Computes the forward pass of LeakyReLU.

        Args:
            input: Input data.
            alpha: Slope of the activation for x < 0.
        """
        # Store alpha as array for consistency in autograd engine
        alpha_arr = np.asarray(alpha)
        ctx.save_for_backward(input, alpha_arr)
        return np.where(input > 0, input, input * alpha)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Computes the gradient of LeakyReLU."""
        input, alpha = ctx.saved_tensors
        grad_input = grad_output * np.where(input > 0, 1.0, alpha)
        return (grad_input, None)
