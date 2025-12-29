from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients, Dim


def _normalize_dim(dim: Optional[Dim] = None) -> tuple[int, ...]:

    if dim is None:
        return None
    elif isinstance(dim, int):
        return (dim,)
    elif isinstance(dim, (tuple, list)):
        return tuple(dim)
    else:
        raise TypeError(f"Invalid dim type {type(dim)}")


@registry_op("sum")
class Sum(Function):
    @staticmethod
    def forward(
        ctx: Context, a: ndarray, dim: Optional[Dim] = None, keepdims: bool = False
    ) -> ndarray:

        dim = _normalize_dim(dim)

        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.saved_shapes = a.shape

        return np.sum(a, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        shape_a = ctx.saved_shapes

        if not ctx.keepdims and ctx.dim is not None:
            grad_output = np.expand_dims(grad_output, ctx.dim)

        grad_output = np.broadcast_to(grad_output, shape_a)

        return (grad_output, None, None)


@registry_op("mean")
class Mean(Function):
    @staticmethod
    def forward(
        ctx: Context, a: ndarray, dim: Optional[Dim] = None, keepdims: bool = False
    ) -> ndarray:

        dim = _normalize_dim(dim)

        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.saved_shapes = a.shape

        if dim is None:
            ctx.N = a.size
        else:
            ctx.N = 1
            for d in dim:
                ctx.N *= a.shape[d]

        return np.mean(a, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        shape_a = ctx.saved_shapes

        if not ctx.keepdims and ctx.dim is not None:
            grad_output = np.expand_dims(grad_output, ctx.dim)

        grad_output = np.broadcast_to(grad_output, shape_a) / ctx.N

        return (grad_output, None, None)


@registry_op("max")
class Max(Function):
    def forward(
        ctx: Context, a: ndarray, dim: Optional[Dim] = None, keepdims: bool = False
    ) -> ndarray:

        dim = _normalize_dim(dim)

        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.saved_shapes = a.shape

        if dim is None:
            ctx.max_vals = a == a.max()
        else:
            ctx.max_vals = a == a.max(axis=dim, keepdims=True)

        return np.max(a, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        shape_a = ctx.saved_shapes

        if not ctx.keepdims and ctx.dim is not None:
            grad_output = np.expand_dims(grad_output, ctx.dim)

        grad_output = np.broadcast_to(grad_output, shape_a)

        grad = grad_output * ctx.max_vals
        num_max = ctx.max_vals.sum(axis=ctx.dim, keepdims=True)
        grad /= num_max

        return (grad, None, None)


@registry_op("min")
class Min(Function):
    def forward(
        ctx: Context, a: ndarray, dim: Optional[Dim] = None, keepdims: bool = False
    ) -> ndarray:

        dim = _normalize_dim(dim)

        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.saved_shapes = a.shape

        if dim is None:
            ctx.min_vals = a == a.min()
        else:
            ctx.min_vals = a == a.min(axis=dim, keepdims=True)

        return np.min(a, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        shape_a = ctx.saved_shapes

        if not ctx.keepdims and ctx.dim is not None:
            grad_output = np.expand_dims(grad_output, ctx.dim)

        grad_output = np.broadcast_to(grad_output, shape_a)

        grad = grad_output * ctx.min_vals
        num_min = ctx.min_vals.sum(axis=ctx.dim, keepdims=True)
        grad /= num_min

        return (grad, None, None)
