from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING
from nova.utils import registry_op
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova._typing import Gradients
    from nova.autograd.engine import Context


@registry_op("inv")
class Inv(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        result = np.linalg.inv(a)
        ctx.save_for_backward(result.T)

        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:

        (inv_T,) = ctx.saved_tensors

        grad_a = -inv_T @ grad_output @ inv_T

        return (grad_a,)
