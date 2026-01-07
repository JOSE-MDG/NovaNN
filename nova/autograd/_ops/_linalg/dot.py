from __future__ import annotations
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("dot")
class Dot(Function):
    """
    Matrix-vector or matrix-matrix dot product.

    Forward: out = input · other
    Backward: ∂L/∂input = grad_output · other^T, ∂L/∂other = input^T · grad_output
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Compute the dot product between input and other."""
        ctx.save_for_backward(input, other)
        return input.dot(other)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for dot product.

        The gradient is: grad_input = grad_output · other^T, grad_other = input^T · grad_output
        """
        input, other = ctx.saved_tensors

        if input.ndim <= 1 and other.ndim <= 1:
            # Vectores
            grad_input = grad_output * other
            grad_other = grad_output * input
        else:
            # Matrices
            grad_input = np.dot(grad_output, other.T)
            grad_other = np.dot(input.T, grad_output)

        return (grad_input, grad_other)
