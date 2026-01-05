from __future__ import annotations
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("clone")
class Clone(Function):
    """
    Clone a tensor.

    Forward: out = copy(input)
    Backward: ∂L/∂input = grad_output
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Return a copy of the input tensor."""
        return input.copy()

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for clone.

        The gradient is: grad_output
        """
        return (grad_output,)
