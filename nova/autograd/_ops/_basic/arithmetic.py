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


def _ensure_array(num: int | float, dtype) -> ndarray:
    """
    Converts scalar to numpy array with specified dtype.

    Args:
        num: Scalar value or array.
        dtype: Target dtype for conversion.

    Returns:
        Input as numpy array with specified dtype.
    """
    is_scalar = isinstance(num, (int, float))
    if is_scalar:
        num_array = np.array(num, dtype=dtype)
    else:
        num_array = num

    return num_array


@registry_op("add")
class Add(Function):
    """
    Element-wise addition with broadcasting support.

    Forward: out = a + b
    Backward: ∂L/∂a = ∂L/∂out, ∂L/∂b = ∂L/∂out
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        """Computes a + b."""
        ctx.saved_shapes = (a.shape, b.shape)
        return a + b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for addition.

        Gradient: ∂(a + b)/∂a = 1, ∂(a + b)/∂b = 1
        Both inputs receive the same gradient, unbroadcasted to original shapes.
        """
        shape_a, shape_b = ctx.saved_shapes
        grad_a = unbroadcasting(grad_output, shape_a)
        grad_b = unbroadcasting(grad_output, shape_b)
        return (grad_a, grad_b)


@registry_op("sub")
class Sub(Function):
    """
    Element-wise subtraction with broadcasting support.

    Forward: out = a - b
    Backward: ∂L/∂a = ∂L/∂out, ∂L/∂b = -∂L/∂out
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        """Computes a - b."""
        ctx.saved_shapes = (a.shape, b.shape)
        return a - b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for subtraction.

        Gradient: ∂(a - b)/∂a = 1, ∂(a - b)/∂b = -1
        First input receives gradient as-is, second receives negated gradient.
        """
        shape_a, shape_b = ctx.saved_shapes
        grad_a = unbroadcasting(grad_output, shape_a)
        grad_b = unbroadcasting(-grad_output, shape_b)
        return (grad_a, grad_b)


@registry_op("mul")
class Mul(Function):
    """
    Element-wise multiplication with broadcasting support.

    Forward: out = a * b
    Backward: ∂L/∂a = ∂L/∂out * b, ∂L/∂b = ∂L/∂out * a
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        """Computes a * b."""
        ctx.save_for_backward(a, b)
        ctx.saved_shapes = (a.shape, b.shape)
        return a * b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for multiplication.

        Gradient: ∂(a * b)/∂a = b, ∂(a * b)/∂b = a
        Each input's gradient is the product of grad_output and the other input.
        """
        a, b = ctx.saved_tensors
        shape_a, shape_b = ctx.saved_shapes

        grad_a = unbroadcasting(grad_output * b, shape_a)
        grad_b = unbroadcasting(grad_output * a, shape_b)

        return (grad_a, grad_b)


@registry_op("div")
class Div(Function):
    """
    Element-wise division with broadcasting support.

    Forward: out = a / b
    Backward: ∂L/∂a = ∂L/∂out * (1/b), ∂L/∂b = ∂L/∂out * (-a/b²)
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        """Computes a / b."""
        ctx.save_for_backward(a, b)
        ctx.saved_shapes = (a.shape, b.shape)
        return a / b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for division.

        Gradient: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
        Uses quotient rule for differentiation.
        """
        a, b = ctx.saved_tensors
        shape_a, shape_b = ctx.saved_shapes

        grad_a = unbroadcasting((1 / b) * grad_output, shape_a)
        grad_b = unbroadcasting((-a / b**2) * grad_output, shape_b)

        return (grad_a, grad_b)


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
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        """Computes a // b."""
        return a // b

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

    Note: Gradient w.r.t. 'b' is complex and not commonly needed,
    so it's simplified to None.
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        """Computes a % b."""
        ctx.saved_shapes = a.shape
        return a % b

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for modulo.

        Gradient: ∂(a % b)/∂a = 1 (simplified)
        The gradient w.r.t. 'a' is 1 in smooth regions.
        """
        shape_a = ctx.saved_shapes
        grad_a = unbroadcasting(grad_output, shape_a)

        return (grad_a, None)


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
    def forward(ctx: Context, a: ndarray, b: ndarray) -> ndarray:
        """Computes floor(a)."""
        return np.floor(a, b)

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
    def forward(ctx: Context, a: ndarray | int, b: ndarray | int) -> ndarray:
        """Computes a^b."""
        b_array = _ensure_array(b, a.dtype)
        result = np.power(a, b_array)
        ctx.save_for_backward(a, b, result)
        ctx.saved_shapes = (a.shape, b_array.shape)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for power operation.

        Gradient: ∂(a^b)/∂a = b * a^(b-1) = b * (a^b)/a
                  ∂(a^b)/∂b = a^b * ln(a)

        Uses mask to handle a ≤ 0 gracefully (sets gradient to 0).
        """
        a, b, result = ctx.saved_tensors
        shape_a, shape_b = ctx.saved_shapes

        mask_valid = a > 0

        # Gradient w.r.t. base: b * a^(b-1)
        grad_a = np.where(
            mask_valid, grad_output * b * result / np.maximum(a, 1e-10), 0.0
        )
        grad_a = unbroadcasting(grad_a, shape_a)

        # Gradient w.r.t. exponent: a^b * ln(a)
        grad_b = np.where(
            mask_valid, grad_output * result * np.log(np.maximum(a, 1e-10)), 0.0
        )
        grad_b = unbroadcasting(grad_b, shape_b)

        return (grad_a, grad_b)


@registry_op("exp")
class Exp(Function):
    """
    Element-wise exponential function.

    Forward: out = e^a
    Backward: ∂L/∂a = ∂L/∂out * e^a

    Note: Gradient is the function value itself (derivative of e^x is e^x).
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray | int) -> ndarray:
        """Computes e^a."""
        result = np.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for exponential.

        Gradient: ∂(e^a)/∂a = e^a
        The derivative of exp is itself.
        """
        (exp,) = ctx.saved_tensors
        grad_a = grad_output * exp
        return (grad_a,)


@registry_op("log")
class Log(Function):
    """
    Element-wise natural logarithm.

    Forward: out = ln(a)
    Backward: ∂L/∂a = ∂L/∂out * (1/a)

    Note: Only defined for a > 0. Uses epsilon for numerical stability.
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray | int) -> ndarray:
        """Computes ln(a)."""
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for natural logarithm.

        Gradient: ∂(ln(a))/∂a = 1/a
        Uses epsilon (1e-20) to prevent division by zero.
        """
        (a,) = ctx.saved_tensors

        grad_a = grad_output * (1 / np.maximum(a, 1e-20))
        return (grad_a,)


@registry_op("sqrt")
class Sqrt(Function):
    """
    Element-wise square root.

    Forward: out = √a
    Backward: ∂L/∂a = ∂L/∂out * (1/(2√a))

    Note: Only defined for a ≥ 0. Uses epsilon for numerical stability.
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        """Computes √a."""
        result = np.sqrt(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for square root.

        Gradient: ∂(√a)/∂a = 1/(2√a)
        Uses epsilon (1e-20) to prevent division by zero.
        """
        (sqrt_a,) = ctx.saved_tensors

        grad_a = grad_output * 0.5 / np.maximum(sqrt_a, 1e-20)
        return (grad_a,)


@registry_op("neg")
class Neg(Function):
    """
    Element-wise negation.

    Forward: out = -a
    Backward: ∂L/∂a = -∂L/∂out
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        """Computes -a."""
        return -a

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

    Note: Sign is a step function with zero gradient everywhere
    except at zero where it's undefined.
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        """Computes sign(a)."""
        return np.sign(a)

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

    Note: Technically undefined at a=0, but we use sign(0)=0 for simplicity.
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        """Computes |a|."""
        ctx.save_for_backward(a)
        return np.abs(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for absolute value.

        Gradient: ∂(|a|)/∂a = sign(a)
        The gradient is +1 where a>0 and -1 where a<0.
        """
        (a,) = ctx.saved_tensors
        grad_a = np.sign(a) * grad_output
        return (grad_a,)


@registry_op("ceil")
class Ceil(Function):
    """
    Element-wise ceiling operation.

    Forward: out = ceil(a)
    Backward: Not differentiable (returns None)

    Note: Ceiling is a step function with zero gradient everywhere
    except at integer boundaries where it's undefined.
    """

    @staticmethod
    def forward(ctx: Context, a: ndarray) -> ndarray:
        """Computes ceil(a)."""
        return np.ceil(a)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Ceiling operation is not differentiable."""
        return (None,)
