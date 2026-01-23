from __future__ import annotations
from nova.autograd.function import Function
from nova.autograd._ops.utils import unbroadcasting
from typing import TYPE_CHECKING, Optional
from numpy import ndarray

if TYPE_CHECKING:
    from nova._typing import Gradients, Size
    from nova.autograd.engine import Context


class ConvMatMul1d(Function):
    """
    Optimized convolution matrix multiplication operation for 1D convolutions.

    This operation fuses the matrix multiplication and reshaping steps of the
    im2col-based convolution algorithm into a single optimized operation.

    Forward:
        out = (weight @ col).reshape(C_out, N, L_out).transpose(1, 0, 2)
        If bias is provided: out = out + bias

    Backward:
        ∂L/∂weight = grad_output @ col^T (reshaped)
        ∂L/∂col = weight^T @ grad_output (reshaped)
        ∂L/∂bias = sum(grad_output, axis=(0, 2))
    """

    @staticmethod
    def forward(
        ctx: Context,
        weight: ndarray,
        bias: Optional[ndarray],
        col: ndarray,
        L_out: int,
        input_size: Size,
    ) -> ndarray:
        """
        Forward pass: Compute convolution via matrix multiplication.

        Performs:
            1. Reshape weight to (C_out, C_in * K)
            2. Matrix multiply: w_col @ col
            3. Reshape and transpose to (N, C_out, L_out)
            4. Add bias if provided
        """
        out_channels = weight.shape[0]
        N = input_size[0]

        w_col = weight.reshape(out_channels, -1)

        out = w_col @ col
        out = out.reshape(out_channels, N, L_out).transpose(1, 0, 2)

        if bias is not None:
            bias_view = bias.view().reshape(1, out_channels, 1)
            out += bias_view
            ctx.bias_shape = bias_view.shape

        ctx.save_for_backward(w_col, col)
        ctx.saved_shapes = weight.shape
        ctx.use_bias = bias is not None
        ctx.out_channels = out_channels

        return out

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for convolution matrix multiplication.

        Gradient:
            ∂L/∂weight: Derived from matrix multiplication rule (A @ B)
                       grad_A = grad_out @ B^T, then reshape to original weight shape

            ∂L/∂col: Derived from matrix multiplication rule (A @ B)
                     grad_B = A^T @ grad_out, then reshape appropriately

            ∂L/∂bias: Sum gradient over batch (N) and spatial (L_out) dimensions
        """
        w_col, col = ctx.saved_tensors
        weight_shape = ctx.saved_shapes

        # 1. Gradient w.r.t. bias
        grad_bias = None
        if ctx.use_bias:
            grad_bias = unbroadcasting(grad_output.sum(axis=(0, 2)), ctx.bias_shape)

        grad_matmul = grad_output.transpose(1, 0, 2).reshape(ctx.out_channels, -1)

        grad_w_col = grad_matmul @ col.T

        # 2. Gradient w.r.t col
        grad_col = w_col.T @ grad_matmul

        # 3. Gradient w.r.t weight
        grad_weight = grad_w_col.reshape(*weight_shape)

        return (grad_weight, grad_bias, grad_col)


class ConvMatMul2d(Function):
    """
    Optimized convolution matrix multiplication operation for 2D convolutions.

    This operation fuses the matrix multiplication and reshaping steps of the
    im2col-based convolution algorithm into a single optimized operation.

    Forward:
        out = (weight @ col).reshape(C_out, N, H_out, W_out).permute(1, 0, 2, 3)
        If bias is provided: out = out + bias

    Backward:
        ∂L/∂weight = grad_output @ col^T (reshaped)
        ∂L/∂col = weight^T @ grad_output (reshaped)
        ∂L/∂bias = sum(grad_output, axis=(0, 2, 3))
    """

    @staticmethod
    def forward(
        ctx: Context,
        weight: ndarray,
        bias: Optional[ndarray],
        col: ndarray,
        H_out: int,
        W_out: int,
        input_size: Size,
    ) -> ndarray:
        """
        Forward pass: Compute convolution via matrix multiplication.

        Performs:
            1. Reshape weight to (C_out, C_in * KH * KW)
            2. Matrix multiply: w_col @ col
            3. Reshape and permute to (N, C_out, H_out, W_out)
            4. Add bias if provided
        """
        out_channels = weight.shape[0]
        N = input_size[0]

        w_col = weight.reshape(out_channels, -1)

        out = w_col @ col
        out = out.reshape(out_channels, N, H_out, W_out).transpose(1, 0, 2, 3)

        if bias is not None:
            bias_view = bias.view().reshape(1, out_channels, 1, 1)
            out += bias_view
            ctx.bias_shape = bias_view.shape

        ctx.save_for_backward(w_col, col)
        ctx.saved_shapes = weight.shape
        ctx.use_bias = bias is not None
        ctx.out_channels = out_channels

        return out

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for convolution matrix multiplication.

        Gradient:
            ∂L/∂weight: Derived from matrix multiplication rule (A @ B)
                       grad_A = grad_out @ B^T, then reshape to original weight shape

            ∂L/∂col: Derived from matrix multiplication rule (A @ B)
                     grad_B = A^T @ grad_out, then reshape appropriately

            ∂L/∂bias: Sum gradient over batch (N) and spatial (H_out, W_out) dimensions
        """
        w_col, col = ctx.saved_tensors
        weight_shape = ctx.saved_shapes

        # 1. Gradient w.r.t. bias
        grad_bias = None
        if ctx.use_bias:
            grad_bias = unbroadcasting(grad_output.sum(axis=(0, 2, 3)), ctx.bias_shape)

        grad_matmul = grad_output.transpose(1, 0, 2, 3).reshape(ctx.out_channels, -1)

        grad_w_col = grad_matmul @ col.T

        # 2. Gradient w.r.t col
        grad_col = w_col.T @ grad_matmul

        # 3. Gradient w.r.t weight
        grad_weight = grad_w_col.reshape(*weight_shape)

        return (grad_weight, grad_bias, grad_col)


class ConvMatMul3d(Function):
    """
    Optimized convolution matrix multiplication operation for 3D convolutions.

    This operation fuses the matrix multiplication and reshaping steps of the
    im2col-based convolution algorithm into a single optimized operation.

    Forward:
        out = (weight @ col).reshape(C_out, N, D_out, H_out, W_out).permute(1, 0, 2, 3, 4)
        If bias is provided: out = out + bias

    Backward:
        ∂L/∂weight = grad_output @ col^T (reshaped)
        ∂L/∂col = weight^T @ grad_output (reshaped)
        ∂L/∂bias = sum(grad_output, axis=(0, 2, 3, 4))
    """

    @staticmethod
    def forward(
        ctx: Context,
        weight: ndarray,
        bias: Optional[ndarray],
        col: ndarray,
        D_out: int,
        H_out: int,
        W_out: int,
        input_size: Size,
    ) -> ndarray:
        """
        Forward pass: Compute convolution via matrix multiplication.

        Performs:
            1. Reshape weight to (C_out, C_in * KD * KH * KW)
            2. Matrix multiply: w_col @ col
            3. Reshape and permute to (N, C_out, D_out, H_out, W_out)
            4. Add bias if provided
        """
        out_channels = weight.shape[0]
        N = input_size[0]

        w_col = weight.reshape(out_channels, -1)

        out = w_col @ col
        out = out.reshape(out_channels, N, D_out, H_out, W_out).transpose(1, 0, 2, 3, 4)

        if bias is not None:
            bias_view = bias.view().reshape(1, out_channels, 1, 1, 1)
            out += bias_view
            ctx.bias_shape = bias_view.shape

        ctx.save_for_backward(w_col, col)
        ctx.saved_shapes = weight.shape
        ctx.use_bias = bias is not None
        ctx.out_channels = out_channels

        return out

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for convolution matrix multiplication.

        Gradient:
            ∂L/∂weight: Derived from matrix multiplication rule (A @ B)
                       grad_A = grad_out @ B^T, then reshape to original weight shape

            ∂L/∂col: Derived from matrix multiplication rule (A @ B)
                     grad_B = A^T @ grad_out, then reshape appropriately

            ∂L/∂bias: Sum gradient over batch (N) and spatial (D_out, H_out, W_out) dimensions
        """
        w_col, col = ctx.saved_tensors
        weight_shape = ctx.saved_shapes

        # 1. Gradient w.r.t. bias
        grad_bias = None
        if ctx.use_bias:
            grad_bias = unbroadcasting(
                grad_output.sum(axis=(0, 2, 3, 4)), ctx.bias_shape
            )

        grad_matmul = grad_output.transpose(1, 0, 2, 3, 4).reshape(ctx.out_channels, -1)

        grad_w_col = grad_matmul @ col.T

        # 2. Gradient w.r.t col
        grad_col = w_col.T @ grad_matmul

        # 3. Gradient w.r.t weight
        grad_weight = grad_w_col.reshape(*weight_shape)

        return (grad_weight, grad_bias, grad_col)
