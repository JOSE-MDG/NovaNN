from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


def _sanitize_index(i):
    if isinstance(i, slice):
        return i

    arr = np.asarray(i)

    if arr.dtype == np.bool_:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.int64)
    elif not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)

    return arr


@registry_op("getitem")
class GetItem(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, idx) -> ndarray:
        if isinstance(idx, tuple):
            actual_idx = tuple(_sanitize_index(i) for i in idx)
        else:
            actual_idx = _sanitize_index(idx)

        ctx.save_for_backward(a)
        ctx.idx = actual_idx

        return a[actual_idx]

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors
        grad_a = np.zeros_like(a, dtype=grad_output.dtype)

        np.add.at(grad_a, ctx.idx, grad_output)

        return (grad_a,)
