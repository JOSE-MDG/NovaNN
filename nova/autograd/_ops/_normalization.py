from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional
from numpy import ndarray
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients

__all__ = ["BatchNorm"]


class BatchNorm(Function):
    """
    Batch Normalization operation with affine transformation.

    Normalizes input across batch and spatial dimensions, then applies
    learnable scale and shift parameters. During training, uses batch
    statistics and updates running averages. During evaluation, uses
    running statistics.

    Forward:
        normalized = (x - μ) / sqrt(σ² + eps)
        output = normalized * weight + bias

        Where μ and σ² are computed over batch and spatial dimensions
        during training, or loaded from running statistics during evaluation.

    Backward:
        ∂L/∂weight = Σ(grad_output * normalized)
        ∂L/∂bias = Σ(grad_output)
        ∂L/∂input = (1/(m*σ)) * [m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)]

        Uses efficient formulation accounting for μ and σ² dependence on input.

    Reduction:
        Running statistics updated with exponential moving average:
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        running_mean: Optional[ndarray],
        running_var: Optional[ndarray],
        weight: Optional[ndarray] = None,
        bias: Optional[ndarray] = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-05,
    ) -> ndarray:
        """
        Compute (x - μ) / sqrt(σ² + eps) * weight + bias.

        Args:
            ctx: Autograd context for saving tensors.
            input: Input tensor of shape (N, C, *). Minimum 2D required.
            running_mean: Running mean of shape (C,). Updated during training.
            running_var: Running variance of shape (C,). Updated during training.
            weight: Optional learnable scale parameter of shape (C,).
            bias: Optional learnable shift parameter of shape (C,).
            training: If True, use batch statistics and update running stats.
            momentum: Momentum for running statistics update.
            eps: Small constant for numerical stability.

        Returns:
            Normalized tensor of same shape as input.

        Raises:
            ValueError: If input has fewer than 2 dimensions.
            ValueError: If running statistics are None in evaluation mode.
        """

        if input.ndim < 2:
            raise ValueError(f"Expected at least 2D input, got {input.ndim}")

        num_features = input.shape[1]
        dims_to_reduce = tuple([0] + list(range(2, input.ndim)))

        if training:
            # Compute batch statistics
            mu = np.mean(input, axis=dims_to_reduce, keepdims=True)
            var_biased = np.var(input, axis=dims_to_reduce, keepdims=True)

            # Calculate number of elements for Bessel's correction
            num_reduced = input.shape[0]
            for d in range(2, input.ndim):
                num_reduced *= input.shape[d]

            # Unbiased variance estimate
            var_unbiased = (
                var_biased * (num_reduced / (num_reduced - 1))
                if num_reduced > 1
                else var_biased
            )

            # Normalize using batch statistics
            std = np.sqrt(var_biased + eps)
            normalized = (input - mu) / std

            # Update running statistics with exponential moving average
            if running_mean is not None and running_var is not None:
                current_mu = mu.reshape(num_features)
                current_var = var_unbiased.reshape(num_features)

                mean_result = (1 - momentum) * running_mean + momentum * current_mu
                var_result = (1 - momentum) * running_var + momentum * current_var
                np.copyto(running_mean, mean_result)
                np.copyto(running_var, var_result)

            # Save for backward
            ctx.std = std
            ctx.training = True

        else:
            # Use running statistics in evaluation mode
            if running_mean is None or running_var is None:
                raise ValueError(
                    "In evaluation mode, running_mean and running_var must be provided."
                )

            mean_shape = [1, num_features] + [1] * (input.ndim - 2)

            mean_broadcast = running_mean.reshape(mean_shape)
            var_broadcast = running_var.reshape(mean_shape)
            std = np.sqrt(var_broadcast + eps)

            normalized = (input - mean_broadcast) / std
            ctx.training = False

        # Apply affine transformation
        output = normalized

        if weight is not None:
            weight_shape = [1, num_features] + [1] * (input.ndim - 2)
            output *= weight.reshape(*weight_shape)

        if bias is not None:
            bias_shape = [1, num_features] + [1] * (input.ndim - 2)
            output += bias.reshape(*bias_shape)

        # Save additional context for backward
        if training:
            ctx.normalized = normalized
            ctx.weight = weight
            ctx.bias = bias
            ctx.eps = eps
            ctx.dims = dims_to_reduce
            ctx.num_reduced = num_reduced

        return output

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Compute gradients for BatchNorm backward pass.

        Gradient:
            ∂L/∂weight = Σ(grad_output * normalized)
            ∂L/∂bias = Σ(grad_output)
            ∂L/∂input = (1/(m*σ)) * [m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)]
        """

        if not ctx.training:
            grad_input = grad_output

            if ctx.weight is not None:
                weight_shape = [1, grad_output.shape[1]] + [1] * (grad_output.ndim - 2)
                grad_input *= ctx.weight.reshape(*weight_shape)

            return (grad_input, None, None, None, None)

        normalized = ctx.normalized
        weight = ctx.weight
        bias = ctx.bias
        std = ctx.std
        dims = ctx.dims
        num_reduced = ctx.num_reduced

        # Gradient w.r.t. bias
        grad_bias = None
        if bias is not None:
            grad_bias = np.sum(grad_output, axis=dims, keepdims=True).reshape(-1)

        # Gradient w.r.t. weight
        grad_weight = None
        if weight is not None:
            grad_weight = np.sum(
                grad_output * normalized, axis=dims, keepdims=True
            ).reshape(-1)

            # Backprop through weight multiplication
            weight_shape = [1, grad_output.shape[1]] + [1] * (grad_output.ndim - 2)
            grad_output = grad_output * weight.reshape(*weight_shape)

        # Gradient w.r.t. input using efficient BatchNorm formula
        grad_normalized = grad_output
        sum_grad = np.sum(grad_normalized, axis=dims, keepdims=True)
        sum_grad_normalized = np.sum(
            grad_normalized * normalized, axis=dims, keepdims=True
        )

        grad_input = (
            (1.0 / num_reduced)
            * (1.0 / std)
            * (
                num_reduced * grad_normalized
                - sum_grad
                - normalized * sum_grad_normalized
            )
        )

        return (
            grad_input,
            None,
            None,
            grad_weight,
            grad_bias,
        )
