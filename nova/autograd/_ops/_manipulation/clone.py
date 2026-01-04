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
    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        return input.copy()

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        return (grad_output,)
