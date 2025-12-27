from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients, Size


@registry_op("stride_tricks")
class AsStrided(Function):
    @staticmethod
    def forward(
        ctx: Context,
        a: ndarray,
        size: Size,
        strides: tuple[int, ...],
    ) -> ndarray:

        if not isinstance(size, tuple):
            size = (size,)
        elif not isinstance(strides, tuple):
            strides = (strides,)

        ctx.saved_shapes = a.shape
        ctx.strides = a.strides
        ctx.out_shape = size
        ctx.out_strides = strides
        ctx.itemsize = a.itemsize

        if len(size) != len(strides):
            raise ValueError(
                f"mismatch with data shapes -> a: {a.shape} - strides: {strides}"
            )

        max_offset = sum((size[i] - 1) * strides[i] for i in range(len(size)))

        if not max_offset < a.nbytes:
            raise ValueError(f"max offsets must be less than num bytes, {a.nbytes}")

        out = np.lib.stride_tricks.as_strided(a, shape=size, strides=strides)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        grad_a = np.zeros(ctx.saved_shapes, dtype=grad_output.dtype)
        coords = np.indices(ctx.out_shape).reshape(len(ctx.out_shape), -1)
        offsets = np.sum(coords.T * np.array(ctx.out_strides), axis=1)
        flat_indices = offsets // ctx.itemsize

        np.add.at(grad_a.ravel(), flat_indices, grad_output.ravel())

        return (grad_a, None, None)
