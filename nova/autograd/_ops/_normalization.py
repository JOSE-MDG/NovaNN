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

            # Pre-allocate statistics
            stat_dims = list(input.shape)
            for dim in dims_to_reduce:
                stat_dims[dim] = 1
            stat_dims = tuple(stat_dims)

            mu = np.empty(stat_dims, dtype=input.dtype)
            var = np.empty(stat_dims, dtype=input.dtype)

            # Compute batch statistics
            np.mean(input, axis=dims_to_reduce, keepdims=True, out=mu)
            np.var(input, axis=dims_to_reduce, keepdims=True, out=var)

            # Calculate number of elements for Bessel's correction
            num_reduced = input.shape[0]
            for d in range(2, input.ndim):
                num_reduced *= input.shape[d]

            # Unbiased variance estimate
            if num_reduced > 1:
                factor = num_reduced / (num_reduced - 1)
                np.multiply(var, factor, out=var)

            # Normalize using batch statistics
            std = np.empty_like(var)
            normalized = np.empty_like(input)

            # Compute std = sqrt(var_biased + eps)
            np.add(var, eps, out=std)
            np.sqrt(std, out=std)

            # Compute normalized = (input - mu) / std
            np.subtract(input, mu, out=normalized)
            np.divide(normalized, std, out=normalized)

            # Update running statistics with exponential moving average
            if running_mean is not None and running_var is not None:
                current_mu = mu.reshape(num_features)
                current_var = var.reshape(num_features)

                running_mean_buf = np.empty_like(running_mean)
                running_var_buf = np.empty_like(running_var)
                temp1 = np.empty_like(current_mu)
                temp2 = np.empty_like(current_var)

                np.subtract(1, momentum, out=running_mean_buf)
                np.subtract(1, momentum, out=running_var_buf)

                np.multiply(running_mean_buf, running_mean, out=running_mean_buf)
                np.multiply(running_var_buf, running_var, out=running_var_buf)

                np.multiply(momentum, current_mu, out=temp1)
                np.multiply(momentum, current_var, out=temp2)

                np.add(running_mean_buf, temp1, out=running_mean)
                np.add(running_var_buf, temp2, out=running_var)

            # Save for backward
            ctx.std = std
            ctx.training = True

        else:
            # Use running statistics in evaluation mode
            if running_mean is None or running_var is None:
                raise ValueError(
                    "In evaluation mode, running_mean and running_var must be provided."
                )

            mean_shape = tuple([1, num_features] + [1] * (input.ndim - 2))

            mean_broadcast = running_mean.reshape(*mean_shape)
            var_broadcast = running_var.reshape(*mean_shape)

            std = np.empty_like(var_broadcast)
            normalized = np.empty_like(input)

            # Compute std = sqrt(var_broadcast + eps)
            np.add(var_broadcast, eps, out=std)
            np.sqrt(std, out=std)

            # Compute normalized = (input - mean_broadcast) / std
            np.subtract(input, mean_broadcast, out=normalized)
            np.divide(normalized, std, out=normalized)

            ctx.training = False

        # Apply affine transformation (use copy so normalized is preserved for backward)
        if weight is not None or bias is not None:
            output = normalized.copy()
        else:
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

        if weight is not None:
            weight_shape = tuple(
                [1, grad_output.shape[1]] + [1] * (grad_output.ndim - 2)
            )

        # Gradient w.r.t. weight
        grad_weight = None
        if weight is not None:
            grad_weight_buf = np.empty_like(grad_output)
            np.multiply(grad_output, normalized, out=grad_weight_buf)
            grad_weight = np.empty(weight_shape, dtype=weight.dtype)
            np.sum(grad_weight_buf, axis=dims, keepdims=True, out=grad_weight)
            grad_weight = grad_weight.reshape(-1)

            # Backprop through weight multiplication
            if weight is not None:
                grad_output *= weight.reshape(*weight_shape)

        # Gradient w.r.t. input using efficient BatchNorm formula
        # Pre-allocate arrays for intermediate computations
        keepdims_shape = [1, grad_output.shape[1]] + [1] * (grad_output.ndim - 2)
        sum_grad = np.empty(keepdims_shape, dtype=grad_output.dtype)
        sum_grad_normalized = np.empty(keepdims_shape, dtype=grad_output.dtype)
        x_hat_sum_term = np.empty_like(grad_output)
        grad_input = np.empty_like(grad_output)
        scale1 = np.empty_like(num_reduced, dtype=grad_output.dtype)
        scale2 = np.empty_like(std)
        scales = np.empty_like(grad_input)

        # Compute sums
        np.sum(grad_output, axis=dims, keepdims=True, out=sum_grad)
        np.multiply(grad_output, normalized, out=x_hat_sum_term)
        np.sum(x_hat_sum_term, axis=dims, keepdims=True, out=sum_grad_normalized)

        # Compute grad_input efficiently
        # grad_input = (1/(m*σ)) * [m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)]
        np.multiply(grad_output, num_reduced, out=grad_input)  # m*dout
        grad_input -= sum_grad  # m*dout - Σ(dout)
        np.multiply(normalized, sum_grad_normalized, out=x_hat_sum_term)
        grad_input -= x_hat_sum_term  # m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)

        np.divide(1.0, num_reduced, out=scale1)
        np.divide(1.0, std, out=scale2)
        np.multiply(scale1, scale2, out=scales)
        np.multiply(grad_input, scales, out=grad_input)

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
        # Normalize shape to tuple so it can be unpacked/iterated consistently
        normalized_shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )
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

        # Apply affine transformation (use copy so normalized is preserved for backward)
        if weight is not None or bias is not None:
            output = normalized.copy()
        else:
            output = normalized

        if weight is not None:
            # Reshape weight to broadcast correctly
            weight_shape = [1] * (input.ndim - ndim) + list(normalized_shape)
            weight_broadcast = weight.reshape(*weight_shape)
            output *= weight_broadcast

        if bias is not None:
            # Reshape bias to broadcast correctly
            bias_shape = [1] * (input.ndim - ndim) + list(normalized_shape)
            bias_broadcast = bias.reshape(*bias_shape)
            output += bias_broadcast

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

        # Reduction axes for parameter gradients: sum over batch dims only (not normalized dims)
        # so grad_bias and grad_weight have shape normalized_shape
        n_batch_dims = len(input_shape) - ndim
        reduce_axes = tuple(range(0, n_batch_dims))

        # Gradient w.r.t. bias
        grad_bias = None
        if bias is not None:
            grad_bias = np.empty(normalized_shape, dtype=bias.dtype)
            np.sum(grad_output, axis=reduce_axes, out=grad_bias)

        # Weight/broadcast shape (for later backprop through weight)
        if weight is not None:
            weight_shape = tuple(
                [1] * (len(input_shape) - ndim) + list(normalized_shape)
            )

        # Gradient w.r.t. weight: Σ(grad_output * normalized) over batch axes
        grad_weight = None
        if weight is not None:
            grad_weight_buf = np.empty_like(grad_output)
            np.multiply(grad_output, normalized, out=grad_weight_buf)
            grad_weight = np.empty(normalized_shape, dtype=weight.dtype)
            np.sum(grad_weight_buf, axis=reduce_axes, out=grad_weight)

        # Backprop through weight multiplication
        if weight is not None:
            weight_broadcast = weight.reshape(*weight_shape)
            grad_output *= weight_broadcast

        # Gradient w.r.t. input using efficient LayerNorm formula
        # Pre-allocate intermediate arrays
        sum_grad = np.empty(keepdims_shape, dtype=grad_output.dtype)
        sum_grad_normalized = np.empty(keepdims_shape, dtype=grad_output.dtype)
        x_hat_sum_term = np.empty_like(grad_output)
        grad_input = np.empty_like(grad_output)
        scale1 = np.empty_like(num_normalized, dtype=grad_output.dtype)
        scale2 = np.empty_like(std)
        scales = np.empty_like(grad_input)

        # Compute sums
        np.sum(grad_output, axis=dims, keepdims=True, out=sum_grad)
        np.multiply(grad_output, normalized, out=x_hat_sum_term)
        np.sum(x_hat_sum_term, axis=dims, keepdims=True, out=sum_grad_normalized)

        # Compute grad_input efficiently
        # grad_input = (1/(m*σ)) * [m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)]
        np.multiply(grad_output, num_normalized, out=grad_input)  # m*dout
        grad_input -= sum_grad  # m*dout - Σ(dout)
        np.multiply(normalized, sum_grad_normalized, out=x_hat_sum_term)
        grad_input -= x_hat_sum_term  # m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)

        np.divide(1.0, num_normalized, out=scale1)
        np.divide(1.0, std, out=scale2)
        np.multiply(scale1, scale2, out=scales)  # 1/(m*σ)
        np.multiply(scales, grad_input, out=grad_input)

        return (
            grad_input,
            None,  # normalized_shape
            grad_weight,
            grad_bias,
            None,  # eps
        )
