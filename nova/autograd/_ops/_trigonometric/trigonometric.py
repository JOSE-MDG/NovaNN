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
    """
    Sine function.

    Forward: out = sin(input)
    Backward: ∂L/∂input = grad_output * cos(input)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the sine of the input."""
        ctx.save_for_backward(input)
        return np.sin(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for sin.

        The gradient is: grad_input = grad_output * cos(input)
        """
        (input,) = ctx.saved_tensors
        grad_input = np.cos(input) * grad_output
        return (grad_input,)


@registry_op("cos")
class Cos(Function):
    """
    Cosine function.

    Forward: out = cos(input)
    Backward: ∂L/∂input = -grad_output * sin(input)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the cosine of the input."""
        ctx.save_for_backward(input)
        return np.cos(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for cos.

        The gradient is: grad_input = -grad_output * sin(input)
        """
        (input,) = ctx.saved_tensors
        grad_input = -np.sin(input) * grad_output
        return (grad_input,)


@registry_op("tan")
class Tan(Function):
    """
    Tangent function.

    Forward: out = tan(input)
    Backward: ∂L/∂input = grad_output * (1 + tan(input)^2)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the tangent of the input."""
        ctx.save_for_backward(input)
        return np.tan(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for tan.

        The gradient is: grad_input = grad_output * (1 + tan(input)^2)
        """
        (input,) = ctx.saved_tensors
        grad_input = (1 + np.tan(input) ** 2) * grad_output
        return (grad_input,)


@registry_op("tanh")
class Tanh(Function):
    """
    Hyperbolic tangent function.

    Forward: out = tanh(input)
    Backward: ∂L/∂input = grad_output * (1 - tanh(input)^2)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the hyperbolic tangent of the input."""
        ctx.save_for_backward(input)
        return np.tanh(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for tanh.

        The gradient is: grad_input = grad_output * (1 - tanh(input)^2)
        """
        (input,) = ctx.saved_tensors
        grad_input = (1 - np.tanh(input) ** 2) * grad_output
        return (grad_input,)


@registry_op("cot")
class Cot(Function):
    """
    Cotangent function.

    Forward: out = 1 / tan(input)
    Backward: ∂L/∂input = -grad_output / sin(input)^2
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the cotangent of the input."""
        ctx.save_for_backward(input)
        return 1 / np.tan(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for cot.

        The gradient is: grad_input = -grad_output / sin(input)^2
        """
        (input,) = ctx.saved_tensors
        grad_input = -grad_output / np.sin(input) ** 2
        return (grad_input,)


@registry_op("sec")
class Sec(Function):
    """
    Secant function.

    Forward: out = 1 / cos(input)
    Backward: ∂L/∂input = grad_output * sec(input) * tan(input)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the secant of the input."""
        out = 1 / np.cos(input)
        ctx.save_for_backward(input)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for sec.

        The gradient is: grad_input = grad_output * sec(input) * tan(input)
        """
        (input,) = ctx.saved_tensors
        out = ctx.out
        grad_input = out * np.tan(input) * grad_output
        return (grad_input,)


@registry_op("arcsin")
class Arcsin(Function):
    """
    Inverse sine function.

    Forward: out = arcsin(input)
    Backward: ∂L/∂input = grad_output / sqrt(1 - input^2)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse sine of the input."""
        input_clamped = np.clip(input, -1.0, 1.0)
        ctx.save_for_backward(input_clamped)
        return np.arcsin(input_clamped)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for arcsin.

        The gradient is: grad_input = grad_output / sqrt(1 - input^2)
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output / np.sqrt(1 - input**2)
        return (grad_input,)


@registry_op("arccos")
class Arccos(Function):
    """
    Inverse cosine function.

    Forward: out = arccos(input)
    Backward: ∂L/∂input = -grad_output / sqrt(1 - input^2)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse cosine of the input."""
        input_clamped = np.clip(input, -1.0, 1.0)
        ctx.save_for_backward(input_clamped)
        return np.arccos(input_clamped)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for arccos.

        The gradient is: grad_input = -grad_output / sqrt(1 - input^2)
        """
        (input,) = ctx.saved_tensors
        grad_input = -grad_output / np.sqrt(1 - input**2)
        return (grad_input,)


@registry_op("arctan")
class Arctan(Function):
    """
    Inverse tangent function.

    Forward: out = arctan(input)
    Backward: ∂L/∂input = grad_output / (1 + input^2)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse tangent of the input."""
        ctx.save_for_backward(input)
        return np.arctan(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for arctan.

        The gradient is: grad_input = grad_output / (1 + input^2)
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output / (1 + input**2)
        return (grad_input,)
