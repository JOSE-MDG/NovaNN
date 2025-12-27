from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients


@registry_op("sin")
class Sin(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        ctx.save_for_backward(a)
        return np.sin(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors

        grad_a = np.cos(a) * grad_output
        return (grad_a,)


@registry_op("cos")
class Cos(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        ctx.save_for_backward(a)
        return np.cos(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors
        grad_a = -np.sin(a) * grad_output
        return (grad_a,)


@registry_op("tan")
class Tan(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        ctx.save_for_backward(a)
        return np.tan(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors

        grad_a = (1 + np.tan(a) ** 2) * grad_output
        return (grad_a,)


@registry_op("tanh")
class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        ctx.save_for_backward(a)
        return np.tanh(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors

        grad_a = (1 - np.tanh(a) ** 2) * grad_output
        return (grad_a,)


@registry_op("cot")
class Cot(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        ctx.save_for_backward(a)
        return 1 / np.tan(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors

        grad_a = -grad_output / np.sin(a) ** 2
        return (grad_a,)


@registry_op("sec")
class Sec(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        out = 1 / np.cos(a)
        ctx.save_for_backward(a)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors
        out = ctx.out

        grad_a = (out * np.tan(a)) * grad_output
        return (grad_a,)


@registry_op("arcsin")
class Arcsin(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        a_clamped = np.clip(a, -1.0, 1.0)
        ctx.save_for_backward(a_clamped)
        return np.arcsin(a_clamped)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors

        grad_a = grad_output / np.sqrt(1 - a**2)
        return (grad_a,)


@registry_op("arccos")
class Arccos(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        a_clamped = np.clip(a, -1.0, 1.0)
        ctx.save_for_backward(a_clamped)
        return np.arccos(a_clamped)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors
        grad_a = -grad_output / np.sqrt(1 - a**2)
        return (grad_a,)


@registry_op("arctan")
class Arctan(Function):
    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        ctx.save_for_backward(a)
        return np.arctan(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        (a,) = ctx.saved_tensors

        grad_a = grad_output / (1 + a**2)
        return (grad_a,)
