from __future__ import annotations
from numpy import ndarray
import numpy as np
from typing import TYPE_CHECKING
from nova.utils import registry_op
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova._typing import Gradients
    from nova.autograd.engine import Context


@registry_op("diag")
class Diag(Function):
    """
    Diagonal of a matrix or vector.

    Forward: out = diag(input)
    Backward: ∂L/∂input = diag(∂L/∂out)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, diagonal: int = 0) -> ndarray:
        """compute the diagonal of a vector or matrix"""
        ctx.diagonal = diagonal
        return np.diag(input, diagonal)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for diagonal

        The gradient is: ∂L/∂input = diag(∂L/∂out)
        """
        diagonal = ctx.diagonal
        return (np.diag(grad_output, diagonal),)
