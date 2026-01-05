from __future__ import annotations
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("matmul")
class MatMul(Function):
    """
    Matrix multiplication using '@' operator.

    Forward: out = input @ other
    Backward: ∂L/∂input = grad_output @ other^T, ∂L/∂other = input^T @ grad_output
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Compute matrix multiplication of input and other."""
        ctx.save_for_backward(input, other)
        return input @ other

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for matmul.

        The gradient is: grad_input = grad_output @ other^T, grad_other = input^T @ grad_output
        """
        input, other = ctx.saved_tensors
        grad_input = grad_output @ other.T
        grad_other = input.T @ grad_output
        return (grad_input, grad_other)
