from __future__ import annotations
from numpy import ndarray
import numpy as np
from typing import TYPE_CHECKING
from nova.utils import registry_op
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova._typing import Gradients
    from nova.autograd.engine import Context


@registry_op("trace")
class Trace(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        ctx.save_for_backward(a)
        return np.linalg.trace(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors

        N = a.shape[0]
        grad_a = np.eye(N=N) * grad_output

        return (grad_a,)
