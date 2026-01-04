from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("sigmoid")
class Sigmoid(Function):
    """
    Sigmoid activation function.

    Forward: out = 1 / (1 + exp(-x))
    Backward: dL/dx = dL/dout * out * (1 - out)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Computes the logistic sigmoid function."""
        output = 1 / (1 + np.exp(-input))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Computes the gradient using the pre-computed forward output."""
        (output,) = ctx.saved_tensors
        grad_input = grad_output * output * (1 - output)
        return (grad_input,)
