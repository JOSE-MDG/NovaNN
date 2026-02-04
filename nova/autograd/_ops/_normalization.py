from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional
from numpy import ndarray
from nova.utils.decorators import no_inplace_op
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients

__all__ = ["BatchNorm", "LayerNorm"]


@no_inplace_op
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

                running_mean[:] = (1 - momentum) * running_mean + momentum * current_mu
                running_var[:] = (1 - momentum) * running_var + momentum * current_var

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
                np.multiply(
                    grad_input, ctx.weight.reshape(*weight_shape), out=grad_input
                )

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
            # Pre-allocate with keepdims shape: [1, C, 1, 1, ...]
            keepdims_shape = [1, grad_output.shape[1]] + [1] * (grad_output.ndim - 2)
            grad_bias = np.empty(keepdims_shape, dtype=bias.dtype)
            np.sum(grad_output, axis=dims, keepdims=True, out=grad_bias)
            grad_bias = grad_bias.reshape(-1)

        weight_shape = [1, grad_output.shape[1]] + [1] * (grad_output.ndim - 2)

        # Gradient w.r.t. weight
        grad_weight = None
        if weight is not None:
            # weight_shape already has the correct keepdims shape
            grad_weight = np.empty(weight_shape, dtype=weight.dtype)
            np.sum(grad_output * normalized, axis=dims, keepdims=True, out=grad_weight)
            grad_weight = grad_weight.reshape(-1)

            # Backprop through weight multiplication
            if weight is not None:
                grad_output *= weight.reshape(*weight_shape)

        # Gradient w.r.t. input using efficient BatchNorm formula
        # Pre-allocate arrays for intermediate computations
        keepdims_shape = [1, grad_output.shape[1]] + [1] * (grad_output.ndim - 2)
        sum_grad = np.empty(keepdims_shape, dtype=grad_output.dtype)
        sum_grad_normalized = np.empty(keepdims_shape, dtype=grad_output.dtype)

        # Compute sums
        np.sum(grad_output, axis=dims, keepdims=True, out=sum_grad)
        np.sum(
            grad_output * normalized, axis=dims, keepdims=True, out=sum_grad_normalized
        )

        # Compute grad_input efficiently
        # grad_input = (1/(m*σ)) * [m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)]
        grad_input = np.empty_like(grad_output)
        np.multiply(grad_output, num_reduced, out=grad_input)  # m*dout
        grad_input -= sum_grad  # m*dout - Σ(dout)
        grad_input -= (
            normalized * sum_grad_normalized
        )  # m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)
        grad_input *= (1.0 / num_reduced) * (1.0 / std)  # Scale by 1/(m*σ)

        return (
            grad_input,
            None,
            None,
            grad_weight,
            grad_bias,
        )


@no_inplace_op
class LayerNorm(Function):
    """
    Layer Normalization operation.

    Normalizes over the last N dimensions specified by normalized_shape.
    Commonly used in Transformer architectures for its independence from
    batch size.

    Forward:
        normalized = (x - μ) / sqrt(σ² + eps)
        output = normalized * weight + bias

        Where μ and σ² are computed over the normalized dimensions.

    Backward:
        ∂L/∂weight = Σ(grad_output * normalized)
        ∂L/∂bias = Σ(grad_output)
        ∂L/∂input = (1/(m*σ)) * [m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)]

        Uses efficient formulation accounting for μ and σ² dependence on input.
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        normalized_shape: tuple,
        weight: Optional[ndarray] = None,
        bias: Optional[ndarray] = None,
        eps: float = 1e-05,
    ) -> ndarray:
        """
        Compute (x - μ) / sqrt(σ² + eps) * weight + bias
        """
        # Compute dimensions to normalize over (last N dimensions)
        ndim = len(normalized_shape)
        dims_to_normalize = tuple(range(-ndim, 0))

        # Calculate number of elements being normalized
        num_normalized = 1
        for dim in dims_to_normalize:
            num_normalized *= input.shape[dim]

        # Pre-allocate buffers for statistics
        stat_shape = list(input.shape)
        for dim in dims_to_normalize:
            stat_shape[dim] = 1
        stat_shape = tuple(stat_shape)

        mean = np.empty(stat_shape, dtype=input.dtype)
        variance = np.empty(stat_shape, dtype=input.dtype)

        # Compute mean
        np.mean(input, axis=dims_to_normalize, keepdims=True, out=mean)

        # Compute variance: E[(x - μ)²]
        # Allocate centered buffer and compute (x - μ)
        centered = np.empty_like(input)
        np.subtract(input, mean, out=centered)

        # Square centered values in-place, then compute mean
        # We reuse the centered buffer to store squared values
        np.square(centered, out=centered)
        np.mean(centered, axis=dims_to_normalize, keepdims=True, out=variance)

        # Compute std = sqrt(variance + eps)
        std = np.empty_like(variance)
        np.add(variance, eps, out=std)
        np.sqrt(std, out=std)

        # Normalize: (x - μ) / σ
        # We need to recompute (x - μ) since we overwrote centered with squared values
        normalized = np.empty_like(input)
        np.subtract(input, mean, out=normalized)
        np.divide(normalized, std, out=normalized)

        if weight is None and bias is None:
            # Save for backward
            ctx.normalized = normalized
            ctx.std = std
            ctx.weight = weight
            ctx.bias = bias
            ctx.dims_to_normalize = dims_to_normalize
            ctx.num_normalized = num_normalized
            ctx.input_shape = input.shape
            ctx.normalized_shape = normalized_shape
            return normalized

        # Apply affine transformation
        output = normalized

        if weight is not None:
            # Reshape weight to broadcast correctly
            weight_shape = [1] * (input.ndim - ndim) + list(normalized_shape)
            weight_broadcast = weight.reshape(weight_shape)
            output = output * weight_broadcast

        if bias is not None:
            # Reshape bias to broadcast correctly
            bias_shape = [1] * (input.ndim - ndim) + list(normalized_shape)
            bias_broadcast = bias.reshape(bias_shape)
            output = output + bias_broadcast

        # Save for backward
        ctx.normalized = normalized
        ctx.std = std
        ctx.weight = weight
        ctx.bias = bias
        ctx.dims_to_normalize = dims_to_normalize
        ctx.num_normalized = num_normalized
        ctx.input_shape = input.shape
        ctx.normalized_shape = normalized_shape

        return output

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Compute gradients for LayerNorm backward pass.

        Gradient:
            ∂L/∂weight = Σ(grad_output * normalized)
            ∂L/∂bias = Σ(grad_output)
            ∂L/∂input = (1/(m*σ)) * [m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)]
        """
        normalized = ctx.normalized
        std = ctx.std
        weight = ctx.weight
        bias = ctx.bias
        dims = ctx.dims_to_normalize
        num_normalized = ctx.num_normalized
        input_shape = ctx.input_shape
        normalized_shape = ctx.normalized_shape

        ndim = len(normalized_shape)

        # Pre-allocate keepdims shape for reductions
        keepdims_shape = list(input_shape)
        for dim in dims:
            keepdims_shape[dim] = 1
        keepdims_shape = tuple(keepdims_shape)

        # Determine reduction axes for parameter gradients (batch + normalized dims)
        reduce_axes = (0,) + dims

        # Gradient w.r.t. bias
        grad_bias = None
        if bias is not None:
            grad_bias = np.sum(grad_output, axis=reduce_axes)
            # Shape: normalized_shape

        # Gradient w.r.t. weight
        grad_weight = None
        if weight is not None:
            grad_weight = np.sum(grad_output * normalized, axis=reduce_axes)
            # Shape: normalized_shape

        # Backprop through weight multiplication
        if weight is not None:
            weight_shape = [1] * (input_shape.__len__() - ndim) + list(normalized_shape)
            weight_broadcast = weight.reshape(weight_shape)
            grad_output *= weight_broadcast

        # Gradient w.r.t. input using efficient LayerNorm formula
        # Pre-allocate intermediate arrays
        sum_grad = np.empty(keepdims_shape, dtype=grad_output.dtype)
        sum_grad_normalized = np.empty(keepdims_shape, dtype=grad_output.dtype)

        # Compute sums
        np.sum(grad_output, axis=dims, keepdims=True, out=sum_grad)
        np.sum(
            grad_output * normalized, axis=dims, keepdims=True, out=sum_grad_normalized
        )

        # Compute grad_input efficiently
        # grad_input = (1/(m*σ)) * [m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)]
        grad_input = np.empty_like(grad_output)
        np.multiply(grad_output, num_normalized, out=grad_input)  # m*dout
        grad_input -= sum_grad  # m*dout - Σ(dout)
        grad_input -= (
            normalized * sum_grad_normalized
        )  # m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)
        grad_input *= (1.0 / num_normalized) * (1.0 / std)  # Scale by 1/(m*σ)

        return (
            grad_input,
            None,  # normalized_shape
            grad_weight,
            grad_bias,
            None,  # eps
        )
