from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from nova.autograd._ops.utils import unbroadcasting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


def _ensure_array(num: int | float, dtype):
    num_is_scalar = isinstance(num, (int, float))
    if num_is_scalar:
        num_array = np.array(num, dtype=dtype)
    else:
        num_array = num

    return num_array


@registry_op("add")
class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        ctx.saved_shapes = (a.shape, b.shape)
        return a + b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
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
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        shape_a, shape_b = ctx.saved_shapes
        grad_a = unbroadcasting(grad_output, shape_a)
        grad_b = unbroadcasting(-grad_output, shape_b)
        return (grad_a, grad_b)


@registry_op("mul")
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        ctx.saved_shapes = (a.shape, b.shape)
        return a * b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        a, b = ctx.saved_tensors
        shape_a, shape_b = ctx.saved_shapes

        grad_a = unbroadcasting(grad_output * b, shape_a)
        grad_b = unbroadcasting(grad_output * a, shape_b)

        return (grad_a, grad_b)


@registry_op("div")
class Div(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        ctx.save_for_backward(a, b)
        ctx.saved_shapes = (a.shape, b.shape)
        return a / b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:

        a, b = ctx.saved_tensors
        shape_a, shape_b = ctx.saved_shapes

        grad_a = unbroadcasting((1 / b) * grad_output, shape_a)
        grad_b = unbroadcasting((-a / b**2) * grad_output, shape_b)

        return (grad_a, grad_b)


@registry_op("divint")
class DivInt(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        return a // b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        return (None, None)


@registry_op("mod")
class Mod(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        ctx.saved_shapes = a.shape
        return a % b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        shape_a = ctx.saved_shapes
        grad_a = unbroadcasting(grad_output, shape_a)

        return (grad_a, None)


@registry_op("floor")
class Floor(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        return np.floor(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        return (None, None)


@registry_op("pow")
class Pow(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray | int, b: ndarray | int) -> ndarray:

        b_array = _ensure_array(b, a.dtype)
        result = np.power(a, b_array)
        ctx.save_for_backward(a, b, result)
        ctx.saved_shapes = (a.shape, b_array.shape)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        a, b, result = ctx.saved_tensors
        shape_a, shape_b = ctx.saved_shapes

        mask_valid = a > 0

        grad_a = np.where(
            mask_valid, grad_output * b * result / np.maximum(a, 1e-10), 0.0
        )
        grad_a = unbroadcasting(grad_a, shape_a)

        grad_b = np.where(
            mask_valid, grad_output * result * np.log(np.maximum(a, 1e-10)), 0.0
        )
        grad_b = unbroadcasting(grad_b, shape_b)

        return (grad_a, grad_b)


@registry_op("exp")
class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray | int) -> ndarray:
        result = np.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (exp,) = ctx.saved_tensors
        grad_a = grad_output * exp
        return (grad_a,)


@registry_op("log")
class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray | int) -> ndarray:
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors

        grad_a = grad_output * (1 / np.maximum(a, 1e-20))
        return (grad_a,)


@registry_op("sqrt")
class Sqrt(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        result = np.sqrt(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (sqrt_a,) = ctx.saved_tensors

        grad_a = grad_output * 0.5 / np.maximum(sqrt_a, 1e-20)
        return (grad_a,)


@registry_op("neg")
class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        return -a

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        return (-grad_output,)


@registry_op("sign")
class Sign(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        return np.sign(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        return (None,)


@registry_op("abs")
class Abs(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        ctx.save_for_backward(a)
        return np.abs(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors
        grad_a = np.sign(a) * grad_output
        return (grad_a,)


@registry_op("ceil")
class Ceil(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        return np.ceil(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        return (None,)
