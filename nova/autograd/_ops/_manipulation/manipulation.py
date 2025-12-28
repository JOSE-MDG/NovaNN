from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients, Dim


@registry_op("permute")
class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, dims: Optional[Dim] = None) -> ndarray:
        ctx.dims = dims
        return np.transpose(a, axes=dims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        if ctx.dims is None:
            return (grad_output.T, None)

        inv_dims = np.argsort(ctx.dims)
        grad = np.transpose(grad_output, inv_dims)
        return (grad, None)


@registry_op("reshape")
class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, *size: Dim) -> ndarray:
        ctx.saved_shapes = a.shape
        return np.reshape(a, shape=size)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:

        org_shape = ctx.saved_shapes

        return (grad_output.reshape(org_shape), None)


@registry_op("squeeze")
class Squeeze(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, dim: Optional[Dim] = None) -> ndarray:
        ctx.saved_shapes = a.shape
        return np.squeeze(a, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        org_shape = ctx.saved_shapes

        return (grad_output.reshape(org_shape), None)


@registry_op("unsqueeze")
class UnSqueeze(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, dim: Dim) -> ndarray:
        ctx.saved_shapes = a.shape
        return np.expand_dims(a, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        org_shape = ctx.saved_shapes

        return (grad_output.reshape(org_shape), None)


@registry_op("stack")
class Stack(Function):
    @staticmethod
    def forward(ctx: Context, inputs: list[ndarray], dim: Dim = 0) -> ndarray:
        ctx.dim = dim
        ctx.N = len(inputs)
        return np.stack(inputs, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:

        grads = np.split(grad_output, ctx.N, axis=ctx.dim)
        grads = [g.squeeze(ctx.dim) for g in grads]
        return (*grads, None)


@registry_op("concat")
class Concat(Function):
    @staticmethod
    def forward(ctx: Context, tensors: list[ndarray], dim: ndarray = 0) -> ndarray:
        ctx.saved_shapes = [t.shape for t in tensors]
        ctx.dim = dim
        return np.concatenate(tensors, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        shapes = ctx.saved_shapes

        sizes = [s[ctx.dim] for s in shapes]
        offsets = np.cumsum(sizes)[:-1]

        grads = np.split(grad_output, offsets, axis=ctx.dim)
        return (*grads, None)


@registry_op("split")
class Split(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, sections: int, dim: Dim = 0) -> ndarray:
        ctx.dim = dim
        return np.array_split(a, sections, dim)

    @staticmethod
    def backward(ctx: Context, *grad_output: ndarray) -> Gradients:

        grads = np.concatenate(grad_output, ctx.dim)
        return (grads, None, None)


@registry_op("clamp")
class Clamp(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, min_val: float, max_val: float) -> ndarray:
        ctx.save_for_backward(a)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return np.clip(a, min_val, max_val)

    @staticmethod
    def backward(ctx: Context, grad_output) -> Gradients:
        (a,) = ctx.saved_tensors

        mask = (a >= ctx.min_val) & (a <= ctx.max_val)
        grad_a = grad_output * mask
        return (grad_a, None, None)


@registry_op("tile")
class Tile(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, repeats: int) -> ndarray:
        ctx.repeats = repeats
        ctx.saved_shapes = a.shape
        return np.tile(a, repeats)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        shape_a = ctx.saved_shapes
        grad = grad_output
        for axis, rep in enumerate(ctx.repeats):
            grad = grad.reshape(rep, shape_a[axis], *shape_a[axis + 1 :]).sum(axis=0)

        return (grad, None)


@registry_op("repeat")
class Repeat(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, repeats: int, dim: Dim = 0) -> ndarray:
        ctx.saved_shapes = a.shape
        ctx.repeats = repeats
        ctx.dim = dim

        return np.repeat(a, repeats, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output) -> Gradients:
        shape_a = ctx.saved_shapes

        if ctx.dim is None:
            grad = grad_output.reshape(-1, ctx.repeats).sum(axis=1)
            grad = grad.reshape(shape_a)
        else:
            grad = grad_output.reshape(
                *shape_a[: ctx.dim],
                shape_a[ctx.dim],
                ctx.repeats,
                *shape_a[ctx.dim + 1 :],
            ).sum(
                axis=ctx.dim + 1
            )  # *shape_a[ctx.dim + 1 :] normaly is a epmty tuple

        return (grad, None, None)


@registry_op("pad")
class Pad(Function):
    @staticmethod
    def forward(
        ctx: Context,
        a: ndarray,
        pad_width: tuple[tuple[int, ...], ...],
        mode: str = "constant",
    ) -> ndarray:
        ctx.pad_width = pad_width
        return np.pad(a, pad_width, mode=mode)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        pad_width = ctx.pad_width

        slices = []

        for i, (before, after) in enumerate(pad_width):

            end = grad_output.shape[i] - after
            slices.append(slice(before, end))

        grad = grad_output[tuple(slices)]
        return (grad, None, None)
