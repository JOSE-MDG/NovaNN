from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd._ops.utils import unbroadcasting
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("maximum")
class Maximum(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        ctx.save_for_backward(a, b)
        ctx.saved_shapes = (a.shape, b.shape)
        return np.maximum(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        a, b = ctx.saved_tensors
        shape_a, shape_b = ctx.saved_shapes

        mask_a = a > b
        mask_b = ~mask_a
        mask_eq = a == b

        grad_a = mask_a + 0.5 * mask_eq
        grad_b = mask_b + 0.5 * mask_eq

        grad_a = unbroadcasting(grad_output * grad_a, shape_a)
        grad_b = unbroadcasting(grad_output * grad_b, shape_b)

        return (grad_a, grad_b)


@registry_op("minimum")
class Minimum(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        ctx.save_for_backward(a, b)
        ctx.saved_shapes = (a.shape, b.shape)
        return np.minimum(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        a, b = ctx.saved_tensors
        shape_a, shape_b = ctx.saved_shapes

        mask_a = a > b
        mask_b = ~mask_a
        mask_eq = a == b

        grad_a = mask_a + 0.5 * mask_eq
        grad_b = mask_b + 0.5 * mask_eq

        grad_a = unbroadcasting(grad_output * grad_a, shape_a)
        grad_b = unbroadcasting(grad_output * grad_b, shape_b)

        return (grad_a, grad_b)


@registry_op("where")
class Where(Function):
    @staticmethod
    def forward(ctx: Context, condition, a: ndarray, b: ndarray) -> ndarray:
        ctx.save_for_backward(condition)
        ctx.saved_shapes = (a.shape, b.shape)
        return np.where(condition, a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (condition,) = ctx.saved_tensors
        shape_a, shape_b = ctx.saved_shapes

        grad_a = unbroadcasting(np.where(condition, grad_output, 0.0), shape_a)
        grad_b = unbroadcasting(np.where(condition, 0.0, grad_output), shape_b)

        return (None, grad_a, grad_b)
