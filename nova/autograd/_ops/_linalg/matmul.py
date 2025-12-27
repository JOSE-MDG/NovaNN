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
    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:

        a, b = ctx.saved_tensors

        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output

        return (grad_a, grad_b)
