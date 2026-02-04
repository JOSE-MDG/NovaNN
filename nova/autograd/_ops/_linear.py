from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional
from numpy import ndarray
from nova.utils.decorators import no_inplace_op
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients

__all__ = ["Dense"]


@no_inplace_op
class Dense(Function):
    """Linear Layer (efficient formulation)

    Forward: X @ W.T + b
    Backward:
        ∂L/∂X = grad_output @ weight
        ∂L/∂W = grad_output.T @ input
        ∂L/∂b = Σ(grad_output)
    """

    @staticmethod
    def forward(
        ctx: Context, input: ndarray, weight: ndarray, bias: Optional[ndarray] = None
    ) -> ndarray:
        """
        Computes Y = X @ W.T + b
        """
        ctx.save_for_backward(input, weight, bias)

        out_shape = list(input.shape)
        out_shape[-1] = weight.shape[0]

        output = np.empty(out_shape, dtype=input.dtype)
        np.matmul(input, weight.T, out=output)

        if bias is not None:
            output += bias
            ctx.bias_shape = bias.shape

        ctx.saved_shapes = (input.shape, weight.shape)

        return output

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Compute backward pass for linear layer

        Gradients:
            ∂L/∂X = grad_output @ weight
            ∂L/∂W = grad_output.T @ input
            ∂L/∂b = Σ(grad_output)
        """
        input, weight, bias = ctx.saved_tensors
        input_shape, weight_shape = ctx.saved_shapes

        grad_input = np.empty(input_shape, dtype=input.dtype)
        grad_weight = np.empty(weight_shape, dtype=weight.dtype)
        grad_bias = (
            np.empty(ctx.bias_shape, dtype=bias.dtype) if bias is not None else None
        )

        # Gradient w.r.t input
        np.matmul(grad_output, weight, out=grad_input)

        # Gradient w.r.t weight
        np.matmul(grad_output.T, input, out=grad_weight)

        # Gradient w.r.t bias
        if bias is not None:
            np.sum(grad_output, axis=0, out=grad_bias)

        return (grad_input, grad_weight, grad_bias)
