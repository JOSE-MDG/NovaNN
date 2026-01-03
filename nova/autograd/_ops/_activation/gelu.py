from __future__ import annotations
import numpy as np
import math
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("gelu")
class GELU(Function):
    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        ctx.save_for_backward(input)
        inner = math.sqrt(2.0 / math.pi) * (input + 0.044715 * np.power(input, 3))
        out = 0.5 * input * (1.0 + np.tanh(inner))
        ctx.save_for_backward(input, inner)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        input, inner = ctx.saved_tensors
        tanh_inner = np.tanh(inner)
        left = 0.5 * (1.0 + tanh_inner)
        right = (
            0.5
            * input
            * (1.0 - np.power(tanh_inner, 2))
            * math.sqrt(2.0 / math.pi)
            * (1.0 + 3.0 * 0.044715 * np.power(input, 2))
        )
        grad_input = grad_output * (left + right)
        return (grad_input,)
