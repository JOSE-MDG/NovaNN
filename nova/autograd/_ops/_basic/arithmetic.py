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


@registry_op("add")
class Add(Function):
    """
    Element-wise addition with broadcasting support.

    Forward: out = a + b
    Backward: ∂L/∂a = ∂L/∂out, ∂L/∂b = ∂L/∂out
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Computes a + b."""
        other_array = np.array(other, dtype=input.dtype)
        ctx.saved_shapes = (input.shape, other_array.shape)
        ctx.save_for_backward(input, other_array)
        return input + other_array

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for addition.

        Gradient: ∂(a + b)/∂a = 1, ∂(a + b)/∂b = 1
        Both inputs receive the same gradient, unbroadcasted to original shapes.
        """
        shape_input, shape_other = ctx.saved_shapes
        grad_input = unbroadcasting(grad_output, shape_input)
        grad_other = unbroadcasting(grad_output, shape_other)
        return (grad_input, grad_other)


@registry_op("sub")
class Sub(Function):
    """
    Element-wise subtraction with broadcasting support.

    Forward: out = a - b
    Backward: ∂L/∂a = ∂L/∂out, ∂L/∂b = -∂L/∂out
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Computes a - b."""
        other_array = np.array(other, dtype=input.dtype)
        ctx.saved_shapes = (input.shape, other_array.shape)
        ctx.save_for_backward(input, other_array)
        return input - other_array

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for subtraction.

        Gradient: ∂(a - b)/∂a = 1, ∂(a - b)/∂b = -1
        First input receives gradient as-is, second receives negated gradient.
        """
        shape_input, shape_other = ctx.saved_shapes
        grad_input = unbroadcasting(grad_output, shape_input)
        grad_other = unbroadcasting(-grad_output, shape_other)
        return (grad_input, grad_other)


@registry_op("mul")
class Mul(Function):
    """
    Element-wise multiplication with broadcasting support.

    Forward: out = a * b
    Backward: ∂L/∂a = ∂L/∂out * b, ∂L/∂b = ∂L/∂out * a
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Computes a * b."""
        other_array = np.array(other, dtype=input.dtype)
        ctx.saved_shapes = (input.shape, other_array.shape)
        ctx.save_for_backward(input, other_array)
        return input * other_array

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for multiplication.

        Gradient: ∂(a * b)/∂a = b, ∂(a * b)/∂b = a
        Each input's gradient is the product of grad_output and the other input.
        """
        input, other = ctx.saved_tensors
        shape_input, shape_other = ctx.saved_shapes

        grad_input = unbroadcasting(grad_output * other, shape_input)
        grad_other = unbroadcasting(grad_output * input, shape_other)

        return (grad_input, grad_other)


@registry_op("div")
class Div(Function):
    """
    Element-wise division with broadcasting support.

    Forward: out = a / b
    Backward: ∂L/∂a = ∂L/∂out * (1/b), ∂L/∂b = ∂L/∂out * (-a/b²)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Computes a / b."""
        other_array = np.array(other, dtype=input.dtype)
        ctx.saved_shapes = (input.shape, other_array.shape)
        ctx.save_for_backward(input, other_array)
        return input / other_array

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for division.

        Gradient: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
        Uses quotient rule for differentiation.
        """
        input, other = ctx.saved_tensors
        shape_input, shape_other = ctx.saved_shapes

        grad_input = unbroadcasting((1 / other) * grad_output, shape_input)
        grad_other = unbroadcasting((-input / other**2) * grad_output, shape_other)

        return (grad_input, grad_other)


@registry_op("divint")
class DivInt(Function):
    """
    Element-wise floor division (integer division).

    Forward: out = a // b
    Backward: Not differentiable (returns None for both gradients)

    Note: Floor division is a discrete operation with zero gradient everywhere
    except at discontinuities where it's undefined.
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Computes a // b."""
        other_array = np.array(other, dtype=input.dtype)
        return input // other_array

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Floor division is not differentiable."""
        return (None, None)


@registry_op("mod")
class Mod(Function):
    """
    Element-wise modulo operation.

    Forward: out = a % b
    Backward: ∂L/∂a = ∂L/∂out, ∂L/∂b = None
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Computes a % b."""
        other_array = np.array(other, dtype=input.dtype)
        ctx.saved_shapes = (input.shape, other_array.shape)
        return input % other_array

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for modulo.

        Gradient: ∂(a % b)/∂a = 1 (simplified)
        The gradient w.r.t. 'a' is 1 in smooth regions.
        """
        shape_input = ctx.saved_shapes
        grad_input = unbroadcasting(grad_output, shape_input)

        return (grad_input, None)


@registry_op("floor")
class Floor(Function):
    """
    Element-wise floor operation.

    Forward: out = floor(a)
    Backward: Not differentiable (returns None)

    Note: Floor is a step function with zero gradient everywhere
    except at integer boundaries where it's undefined.
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, other: ndarray) -> ndarray:
        """Computes floor(a)."""
        other_array = np.array(other, dtype=input.dtype)
        return np.floor(input, other_array)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Floor operation is not differentiable."""
        return (None, None)


@registry_op("pow")
class Pow(Function):
    """
    Element-wise power operation.

    Forward: out = a^b
    Backward: ∂L/∂a = ∂L/∂out * b * a^(b-1)
              ∂L/∂b = ∂L/∂out * a^b * ln(a)

    Note: Only defined for a > 0 to avoid complex numbers.
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray | int, other: ndarray | int) -> ndarray:
        """Computes a^b."""
        other_array = np.array(other, dtype=input.dtype)
        result = np.power(input, other_array)
        ctx.save_for_backward(input, other, result)
        ctx.saved_shapes = (input.shape, other_array.shape)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for power operation.

        Gradient: ∂(a^b)/∂a = b * a^(b-1) = b * (a^b)/a
                  ∂(a^b)/∂b = a^b * ln(a)

        Uses mask to handle a ≤ 0 gracefully (sets gradient to 0).
        """
        input, other, result = ctx.saved_tensors
        shape_input, shape_other = ctx.saved_shapes

        mask_valid = input > 0

        grad_input = np.where(
            mask_valid, grad_output * other * result / np.maximum(input, 1e-10), 0.0
        )
        grad_input = unbroadcasting(grad_input, shape_input)

        grad_other = np.where(
            mask_valid, grad_output * result * np.log(np.maximum(input, 1e-10)), 0.0
        )
        grad_other = unbroadcasting(grad_other, shape_other)

        return (grad_input, grad_other)


@registry_op("exp")
class Exp(Function):
    """
    Element-wise exponential function.

    Forward: out = e^a
    Backward: ∂L/∂a = ∂L/∂out * e^a

    Note: Gradient is the function value itself (derivative of e^x is e^x).
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray | int) -> ndarray:
        """Computes e^a."""
        result = np.exp(input)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for exponential.

        Gradient: ∂(e^a)/∂a = e^a
        The derivative of exp is itself.
        """
        (exp_result,) = ctx.saved_tensors
        grad_input = grad_output * exp_result
        return (grad_input,)


@registry_op("log")
class Log(Function):
    """
    Element-wise natural logarithm.

    Forward: out = ln(a)
    Backward: ∂L/∂a = ∂L/∂out * (1/a)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray | int) -> ndarray:
        """Computes ln(a)."""
        ctx.save_for_backward(input)
        return np.log(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for natural logarithm.

        Gradient: ∂(ln(a))/∂a = 1/a
        Uses epsilon (1e-20) to prevent division by zero.
        """
        (input,) = ctx.saved_tensors

        grad_input = grad_output * 1 / (input + 1e-20)
        return (grad_input,)


@registry_op("sqrt")
class Sqrt(Function):
    """
    Element-wise square root.

    Forward: out = √a
    Backward: ∂L/∂a = ∂L/∂out * (1/(2√a))
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Computes √a."""
        result = np.sqrt(input)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for square root.

        Gradient: ∂(√a)/∂a = 1/(2√a)
        Uses epsilon (1e-20) to prevent division by zero.
        """
        (sqrt_input,) = ctx.saved_tensors

        grad_input = grad_output * 0.5 / (sqrt_input + 1e-20)
        return (grad_input,)


@registry_op("neg")
class Neg(Function):
    """
    Element-wise negation.

    Forward: out = -a
    Backward: ∂L/∂a = -∂L/∂out
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Computes -a."""
        return -input

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for negation.

        Gradient: ∂(-a)/∂a = -1
        Simply negates the incoming gradient.
        """
        return (-grad_output,)


@registry_op("sign")
class Sign(Function):
    """
    Element-wise sign function.

    Forward: out = sign(a) = {-1 if a<0, 0 if a=0, 1 if a>0}
    Backward: Not differentiable (returns None)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Computes sign(a)."""
        return np.sign(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Sign function is not differentiable."""
        return (None,)


@registry_op("abs")
class Abs(Function):
    """
    Element-wise absolute value.

    Forward: out = |a|
    Backward: ∂L/∂a = ∂L/∂out * sign(a)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Computes |a|."""
        ctx.save_for_backward(input)
        return np.abs(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for absolute value.

        Gradient: ∂(|a|)/∂a = sign(a)
        The gradient is +1 where a>0 and -1 where a<0.
        """
        (input,) = ctx.saved_tensors
        grad_input = np.sign(input) * grad_output
        return (grad_input,)


@registry_op("ceil")
class Ceil(Function):
    """
    Element-wise ceiling operation.

    Forward: out = ceil(a)
    Backward: Not differentiable (returns None)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Computes ceil(a)."""
        return np.ceil(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Ceiling operation is not differentiable."""
        return (None,)
