from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from nova._typing import Size, Dim

__all__ = ["unbroadcasting"]


def unbroadcasting(grad: ndarray, shape: Size) -> ndarray:
    """
    Reduce a broadcasted gradient back to the original tensor shape.
    """
    # Remove extra leading dimensions
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Find axes where broadcasting occurred (size 1 in original)
    axes_to_sum = []
    for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            axes_to_sum.append(i)

    # Sum all at once to avoid index shifting
    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)

    return grad


def ensure_casting(dest: ndarray, src: ndarray) -> tuple[ndarray, ndarray]:
    """
    Ensures that the source array allows safe casting to the destination dtype.

    If the dtypes differ, this function attempts to cast the source array to match
    the destination's dtype. This is critical for in-place operations to prevent
    type mismatch errors when copying data.

    Args:
        dest (ndarray): The destination array (the 'out' buffer).
        src (ndarray): The source array containing the computed result.

    Returns:
        Tuple[ndarray, ndarray]: A tuple containing (dest, src). Note that 'src'
        might be a new copy if casting was necessary.
    """
    if dest.dtype != src.dtype:
        src = src.astype(dest.dtype)
    return dest, src


def write_to_buffer(dest: ndarray, src: ndarray) -> ndarray:
    """
    Copies data from the source array into the destination buffer in-place.

    This function handles the low-level memory copy operation. It ensures
    that the data from 'src' is physically written into the memory location
    referenced by 'dest'.

    Args:
        dest (ndarray): The mutable destination array (must not be read-only).
        src (ndarray): The source array to copy from.

    Returns:
        ndarray: The destination array 'dest' after the update.
    """
    dest, src = ensure_casting(dest, src)
    np.copyto(dest, src)
    return dest


def dispatch_output(destination: Optional[ndarray], src: ndarray) -> ndarray:
    """
    Routes the computation result to the correct output destination.

    This function acts as a dispatcher for operations that support an optional
    'out' parameter. If a destination buffer is provided, the result is copied
    into it in-place. Otherwise, the new result array is returned directly.

    Args:
        destination (Optional[ndarray]): The output buffer provided by the user.
            If None, the function simply passes 'src' through.
        src (ndarray): The computed result of the operation.

    Returns:
        ndarray: The 'destination' array if provided (populated with data),
        otherwise the 'src' array.
    """
    if destination is not None:
        return write_to_buffer(destination, src)

    return src


def accelerated_conv_backward(
    weight_shape: Size, grad_output: ndarray, col: ndarray, w_col: ndarray, dims: Dim
) -> tuple[ndarray, ndarray]:
    """
    Optimized backward pass for convolutional layers using matrix multiplication.

    This function computes gradients for both weights and columns in a single,
    memory-efficient pass. It leverages pre-allocated buffers and ensures
    memory contiguity to maximize BLAS performance.

    Args:
        weight_shape (Size): The original shape of the weights (C_out, C_in, ...).
            Used to allocate the properly sized buffer for weight gradients.
        grad_output (ndarray): The gradient of the loss with respect to the
            layer's output. Expected shape is (N, C_out, L_out).
        col (ndarray): The im2col-transformed input matrix used during the
            forward pass. Shape: (C_in * K, N * L_out).
        w_col (ndarray): The weight matrix used during the forward pass,
            reshaped to (C_out, C_in * K).
        dims (Dim): A tuple of axes for the transpose operation to align
            'grad_output' for matrix multiplication.

    Returns:
        tuple[ndarray, ndarray]: A tuple containing:
            - grad_weight: Gradient w.r.t. weights, with shape 'weight_shape'.
            - grad_col: Gradient w.r.t. the 'col' matrix, with shape 'col.shape'.

    Note:
        This function uses 'np.ascontiguousarray' to ensure that the transposed
        output gradients are stored sequentially in memory, enabling SIMD
        vectorization during the 'dot' product.
    """
    dtype = grad_output.dtype
    out_channels = weight_shape[0]

    # Force memory contiguity after transpose to optimize the subsequent GEMM (dot)
    grad_matmul = np.ascontiguousarray(grad_output.transpose(*dims)).reshape(
        out_channels, -1
    )

    # Gradient w.r.t weight: grad_matmul @ col.T
    grad_weight = np.empty(weight_shape, dtype=dtype)
    grad_w_col_view = grad_weight.reshape(out_channels, -1)
    grad_matmul.dot(col.T, out=grad_w_col_view)

    # Gradient w.r.t col: w_col.T @ grad_matmul
    grad_col = np.empty(col.shape, dtype=dtype)
    w_col.T.dot(grad_matmul, out=grad_col)

    return (grad_weight, grad_col)
