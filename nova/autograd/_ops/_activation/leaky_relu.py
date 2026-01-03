from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


def _ensure_array(num: int | float, dtype):
    num_is_scalar = isinstance(num, (int, float))
    if num_is_scalar:
        num_array = np.array(num, dtype=dtype)
    else:
        num_array = num

    return num_array


@registry_op("leaky_relu")
class LeakyReLU(Function):
    @staticmethod
    def forward(ctx: Context, input: ndarray, alpha: float = 0.01) -> ndarray:
        alpha_arra = _ensure_array(alpha)
        ctx.save_for_backward(input, alpha_arra)
        return np.where(input > 0, input, input * alpha)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        input, alpha = ctx.saved_tensors
        grad_input = grad_output * np.where(input > 0, 1.0, alpha)
        return (grad_input, None)
