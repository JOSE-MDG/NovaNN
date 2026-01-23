from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients

__all__ = [
    "Sin",
    "Cos",
    "Tan",
    "Cot",
    "Sec",
    "Csc",
    "Arcsin",
    "Arccos",
    "Arctan",
    "Arccot",
    "Arcsec",
    "Arccsc",
    "Sinh",
    "Cosh",
    "Tanh",
    "Asinh",
    "Acosh",
    "Atanh",
    "Atan2",
]


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


@registry_op("sinh")
class Sinh(Function):
    """
    Hyperbolic sine function.

    Forward: out = sinh(input)
    Backward: ∂L/∂input = grad_output * cosh(input)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the hyperbolic sine of the input."""
        ctx.save_for_backward(input)
        return np.sinh(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for sinh.

        The gradient is: grad_input = grad_output * cosh(input)
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output * np.cosh(input)
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


@registry_op("cosh")
class Cosh(Function):
    """
    Hyperbolic cosine function.

    Forward: out = cosh(input)
    Backward: ∂L/∂input = grad_output * sinh(input)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the hyperbolic cosine of the input."""
        ctx.save_for_backward(input)
        return np.cosh(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for cosh.

        The gradient is: grad_input = grad_output * sinh(input)
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output * np.sinh(input)
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


@registry_op("csc")
class Csc(Function):
    """
    Cosecant function.

    Forward: out = 1 / sin(input)
    Backward: ∂L/∂input = -grad_output * csc(input) * cot(input)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the cosecant of the input."""
        out = 1 / np.sin(input)
        ctx.save_for_backward(input)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for csc.

        The gradient is: grad_input = -grad_output * csc(input) * cot(input)
        """
        (input,) = ctx.saved_tensors
        # csc(x) = ctx.out, cot(x) = 1/tan(x)
        grad_input = -ctx.out * (1 / np.tan(input)) * grad_output
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


@registry_op("atan2")
class Atan2(Function):
    """
    Element-wise inverse tangent of y/x choosing the quadrant correctly.

    Forward: out = arctan2(y, x)
    Backward: ∂L/∂y = grad_output * x / (x^2 + y^2)
              ∂L/∂x = grad_output * -y / (x^2 + y^2)
    """

    @staticmethod
    def forward(ctx: Context, y: ndarray, x: ndarray) -> ndarray:
        """Compute the arctan2 of y and x."""
        ctx.save_for_backward(y, x)
        return np.arctan2(y, x)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for atan2.

        The gradients are:
            grad_y = grad_output * x / (x^2 + y^2)
            grad_x = grad_output * -y / (x^2 + y^2)
        """
        y, x = ctx.saved_tensors
        denom = x**2 + y**2
        grad_y = grad_output * (x / denom)
        grad_x = grad_output * (-y / denom)
        return grad_y, grad_x


@registry_op("asinh")
class Asinh(Function):
    """
    Inverse hyperbolic sine function.

    Forward: out = asinh(input)
    Backward: ∂L/∂input = grad_output / sqrt(input^2 + 1)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse hyperbolic sine of the input."""
        ctx.save_for_backward(input)
        return np.arcsinh(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for asinh.
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output / np.sqrt(input**2 + 1)
        return (grad_input,)


@registry_op("acosh")
class Acosh(Function):
    """
    Inverse hyperbolic cosine function.

    Forward: out = acosh(input)
    Backward: ∂L/∂input = grad_output / sqrt(input^2 - 1)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse hyperbolic cosine of the input."""
        # Dominio: x >= 1
        input_clamped = np.maximum(input, 1.0 + 1e-7)
        ctx.save_for_backward(input_clamped)
        return np.arccosh(input_clamped)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for acosh.
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output / np.sqrt(input**2 - 1)
        return (grad_input,)


@registry_op("atanh")
class Atanh(Function):
    """
    Inverse hyperbolic tangent function.

    Forward: out = atanh(input)
    Backward: ∂L/∂input = grad_output / (1 - input^2)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse hyperbolic tangent of the input."""
        # Dominio: -1 < x < 1
        input_clamped = np.clip(input, -1.0 + 1e-7, 1.0 - 1e-7)
        ctx.save_for_backward(input_clamped)
        return np.arctanh(input_clamped)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for atanh.
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output / (1 - input**2)
        return (grad_input,)


@registry_op("arccot")
class Arccot(Function):
    """
    Inverse cotangent function.

    Forward: out = arctan(1 / input)
    Backward: ∂L/∂input = -grad_output / (1 + input^2)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse cotangent of the input."""
        ctx.save_for_backward(input)
        return np.arctan(1 / input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for arccot.
        """
        (input,) = ctx.saved_tensors
        grad_input = -grad_output / (1 + input**2)
        return (grad_input,)


@registry_op("arcsec")
class Arcsec(Function):
    """
    Inverse secant function.

    Forward: out = arccos(1 / input)
    Backward: ∂L/∂input = grad_output / (|input| * sqrt(input^2 - 1))
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse secant of the input."""
        # Dominio: |input| >= 1
        ctx.save_for_backward(input)
        return np.arccos(1 / input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for arcsec.
        """
        (input,) = ctx.saved_tensors
        abs_input = np.abs(input)
        grad_input = grad_output / (abs_input * np.sqrt(input**2 - 1))
        return (grad_input,)


@registry_op("arccsc")
class Arccsc(Function):
    """
    Inverse cosecante function.

    Forward: out = arcsin(1 / input)
    Backward: ∂L/∂input = -grad_output / (|input| * sqrt(input^2 - 1))
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse cosecant of the input."""
        # Dominio: |input| >= 1
        ctx.save_for_backward(input)
        return np.arcsin(1 / input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for arccsc.
        """
        (input,) = ctx.saved_tensors
        abs_input = np.abs(input)
        grad_input = -grad_output / (abs_input * np.sqrt(input**2 - 1))
        return (grad_input,)
