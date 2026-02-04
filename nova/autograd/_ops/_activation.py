from __future__ import annotations
import math
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.autograd._ops.utils import dispatch_output
from nova.utils import registry_op
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients

__all__ = ["ReLU", "LeakyReLU", "PReLU", "Sigmoid", "GELU"]


@registry_op("gelu")
class GELU(Function):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Forward: out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Backward: ∂L/∂out = 0.5 * input * (1.0 - tanh_inner^2) * sqrt(2.0/pi) * (1.0 + 3.0 * 0.044715 * input^2)
    """

    @staticmethod
    def forward(
        ctx: Context, input: ndarray, _out: Optional[ndarray] = None
    ) -> ndarray:
        """Computes the forward pass using the tanh approximation."""
        k1 = math.sqrt(2.0 / math.pi)
        k2 = 0.044715

        cube = input * input * input
        inner = k1 * (input + k2 * cube)
        tanh_inner = np.tanh(inner)

        output = np.add(tanh_inner, 1.0)
        np.multiply(output, input, out=output)
        np.multiply(output, 0.5, out=output)

        ctx.save_for_backward(input, inner)
        return dispatch_output(_out, output)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Computes the gradient via the chain rule on the approximation."""
        input, inner = ctx.saved_tensors
        tanh_inner = np.tanh(inner)

        left = 0.5 * (1.0 + tanh_inner)

        tanh_sq = np.square(tanh_inner)
        input_sq = np.square(input)

        right = (
            0.5
            * input
            * (1.0 - tanh_sq)
            * math.sqrt(2.0 / math.pi)
            * (1.0 + 3.0 * 0.044715 * input_sq)
        )
        grad_input = grad_output * (left + right)
        return (grad_input,)


@registry_op("leaky_relu")
class LeakyReLU(Function):
    """
    Leaky Rectified Linear Unit activation function.

    Forward: out = x if x > 0 else alpha * x
    Backward: ∂L/∂x = ∂L/∂out * (1 if x > 0 else alpha)
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        alpha: float = 0.01,
        _out: Optional[ndarray] = None,
    ) -> ndarray:
        """
        Computes the forward pass of LeakyReLU.

        Args:
            input: Input data.
            alpha: Slope of the activation for x < 0.
        """
        # Store alpha as array for consistency in autograd engine
        alpha_arr = np.asarray(alpha)
        ctx.save_for_backward(input, alpha_arr)
        output = np.where(input > 0, input, input * alpha)
        return dispatch_output(_out, output)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Computes the gradient of LeakyReLU."""
        input, alpha = ctx.saved_tensors
        grad_input = grad_output * np.where(input > 0, 1.0, alpha)
        return (grad_input, None)


@registry_op("prelu")
class PReLU(Function):
    """
    Parametric Rectified Linear Unit activation function.

    Forward: out = max(0, x) + weight * min(0, x)
    Backward: Gradient is computed for both input and the learnable weight.
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        weight: float | ndarray,
        _out: Optional[ndarray] = None,
    ) -> ndarray:
        """
        Computes the forward pass of PReLU.

        Args:
            input: Input data.
            weight: Learnable parameter (scalar or array).
        """
        weight_arr = np.asarray(weight)
        ctx.save_for_backward(input, weight_arr)
        output = np.where(input > 0, input, weight_arr * input)
        return dispatch_output(_out, output)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Computes gradients for input and weight."""
        input, weight = ctx.saved_tensors

        grad_input = grad_output * np.where(input > 0, 1.0, weight)
        grad_weight = np.sum(grad_output * np.where(input > 0, 0.0, input))

        return (grad_input, grad_weight)


@registry_op("relu")
class ReLU(Function):
    """
    Rectified Linear Unit activation function.

    Forward: out = max(0, x)
    Backward: ∂L/∂x = ∂L/∂out * (1 if x > 0 else 0)
    """

    @staticmethod
    def forward(
        ctx: Context, input: ndarray, _out: Optional[ndarray] = None
    ) -> ndarray:
        """Computes the forward pass of ReLU."""
        ctx.save_for_backward(input)
        output = np.maximum(input, 0, out=_out)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Computes the gradient of ReLU."""
        (input,) = ctx.saved_tensors
        grad_input = grad_output * (input > 0)
        return (grad_input,)


@registry_op("sigmoid")
class Sigmoid(Function):
    """
    Sigmoid activation function.

    Forward: out = 1 / (1 + exp(-x))
    Backward: ∂L/∂x = ∂L/∂out * out * (1 - out)
    """

    @staticmethod
    def forward(
        ctx: Context, input: ndarray, _out: Optional[ndarray] = None
    ) -> ndarray:
        """Computes the logistic sigmoid function."""
        neg_input = np.negative(input)
        exp_val = np.exp(neg_input)
        output = np.add(1.0, exp_val)
        np.divide(1.0, output, out=output)

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """Computes the gradient using the pre-computed forward output."""
        (output,) = ctx.saved_tensors
        grad_input = grad_output * output * (1 - output)
        return (grad_input,)
