from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.utils import registry_op
from nova.utils.decorators import no_inplace_op
from nova.autograd.function import Function
from nova.autograd._ops.utils import unbroadcasting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients, Size

__all__ = ["AsStrided", "View", "Extend"]


@no_inplace_op
@registry_op("stride_tricks")
class AsStrided(Function):
    """
    Create a view of the input array with given shape and strides.

    Forward: out = as_strided(a, shape=size, strides=strides)
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        size: Size,
        strides: tuple[int, ...],
    ) -> ndarray:

        if not isinstance(size, tuple):
            size = (size,)
        elif not isinstance(strides, tuple):
            strides = (strides,)

        ctx.saved_shapes = input.shape
        ctx.strides = input.strides
        ctx.out_shape = size
        ctx.out_strides = strides
        ctx.itemsize = input.itemsize

        if len(size) != len(strides):
            raise ValueError(
                f"mismatch with data shapes -> a: {input.shape} - strides: {strides}"
            )

        max_offset = sum((size[i] - 1) * strides[i] for i in range(len(size)))

        if not max_offset < input.nbytes:
            raise ValueError(f"max offsets must be less than num bytes, {input.nbytes}")

        return np.lib.stride_tricks.as_strided(input, shape=size, strides=strides)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for as_strided.

        The gradient is accumulated into the original tensor positions
        according to the stride offsets.
        """
        grad_input = np.zeros(ctx.saved_shapes, dtype=grad_output.dtype)

        coords = np.indices(ctx.out_shape).reshape(len(ctx.out_shape), -1)
        offsets = np.sum(coords.T * np.array(ctx.out_strides), axis=1)
        flat_indices = offsets // ctx.itemsize

        np.add.at(grad_input.ravel(), flat_indices, grad_output.ravel())

        return (grad_input,)


@no_inplace_op
@registry_op("extend")
class Extend(Function):
    """
    Broadcast an array to a new shape.

    Forward: out = broadcast_to(a, size)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, *size: Size) -> ndarray:
        ctx.saved_shapes = input.shape
        return np.broadcast_to(input, shape=size)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for extend.

        The gradient is reduced back to the original shape
        by summing over broadcasted dimensions.
        """
        org_shape = ctx.saved_shapes
        grad_output = unbroadcasting(grad_output, org_shape)
        return (grad_output,)


@no_inplace_op
@registry_op("view")
class View(Function):
    """
    Reshape an array without copying data.

    Forward: out = reshape(a, size)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, *size: Size) -> ndarray:
        ctx.saved_shapes = input.shape
        return np.reshape(input, size)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for view.

        The gradient is reshaped back to the original input shape.
        """
        org_shape = ctx.saved_shapes
        return (grad_output.reshape(org_shape),)
