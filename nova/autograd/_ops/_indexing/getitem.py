from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("getitem")
class GetItem(Function):
    @staticmethod
    def forward(
        ctx: Context, a: ndarray, idx: tuple[int, ...] | int | ndarray
    ) -> ndarray:
        # ensure that the was a intege

        if isinstance(idx, ndarray):
            if idx.dtype != int:
                idx = idx.astype(int)

        ctx.save_for_backward(a)
        ctx.idx = idx
        return a[idx]

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors

        grad_a = np.zeros_like(a, dtype=grad_output.dtype)

        np.add.at(grad_a, ctx.idx, grad_output)

        return (grad_a, None)
