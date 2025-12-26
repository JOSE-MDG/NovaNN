from __future__ import annotations
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from nova.autograd._ops.utils import unbroadcasting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context


@registry_op("add")
class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        ctx.saved_shapes = (a.shape, b.shape)
        return a + b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> tuple[ndarray | None, ...]:
        shape_a, shape_b = ctx.saved_shapes
        grad_a = unbroadcasting(grad_output, shape_a)
        grad_b = unbroadcasting(grad_output, shape_b)
        return (grad_a, grad_b)


@registry_op("sub")
class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        ctx.saved_shapes = (a.shape, b.shape)
        return a - b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> tuple[ndarray | None, ...]:
        shape_a, shape_b = ctx.saved_shapes
        grad_a = unbroadcasting(grad_output, shape_a)
        grad_b = unbroadcasting(-grad_output, shape_b)
        return (grad_a, grad_b)
