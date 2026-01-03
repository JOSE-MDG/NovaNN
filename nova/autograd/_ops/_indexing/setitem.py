from __future__ import annotations
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("setitem")
class SetItem(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, key: Any, value: Any) -> ndarray:
        ctx.save_for_backward(key)
        a[key] = value
        return a

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (key,) = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[key] = 0.0
        return (grad_input,)
