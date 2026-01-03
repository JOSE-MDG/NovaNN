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


@registry_op("prelu")
class PReLU(Function):
    @staticmethod
    def forward(ctx: Context, input: ndarray, weight: float | ndarray) -> ndarray:
        weight_array = _ensure_array(weight, input.dtype)
        ctx.save_for_backward(input, weight_array)
        return np.where(input > 0, input, weight * input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        input, weight = ctx.saved_tensors
        grad_input = grad_output * np.where(input > 0, 1.0, weight)
        grad_weight = np.sum(grad_output * np.where(input > 0, 0.0, input))
        return (grad_input, grad_weight)
