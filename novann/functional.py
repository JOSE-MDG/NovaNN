import numpy as np
from numpy.lib.stride_tricks import as_strided
from novann.module import Parameters
from novann._typing import Shape, KernelSize, Padding, Stride, ArrayAndExtras
from typing import Optional, Tuple


def relu(input: np.ndarray, extras: bool = False) -> np.ndarray:
    """Applies the ReLU activation function element-wise.

    Args:
        input (np.ndarray): Input tensor.
        extras (bool): If True, returns both output and mask used for backward pass.

    Returns:
        np.ndarray: Activated tensor or tuple (output, mask) if extras=True.
    """
    mask = (input > 0).astype(input.dtype, copy=False)
    out = input * mask
    if extras:
        return out, mask
    return out


def leaky_relu(input: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
    """Applies the Leaky ReLU activation function element-wise.

    Args:
        input (np.ndarray): Input tensor.
        negative_slope (float): Slope for negative input values. Default: 0.01.

    Returns:
        np.ndarray: Activated tensor.
    """
    return np.where(input >= 0, input, negative_slope * input).astype(
        input.dtype, copy=False
    )


def sigmoid(input: np.ndarray) -> np.ndarray:
    """Applies the sigmoid activation function element-wise.

    Args:
        input (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Tensor with sigmoid applied.
    """
    return (1 / (1 + np.exp(-input))).astype(input.dtype, copy=False)


def softmax(input: np.ndarray, dim: int = 1) -> np.ndarray:
    """Computes the softmax along a specified dimension.

    Args:
        input (np.ndarray): Input tensor.
        dim (int): Axis along which softmax is computed. Default: 1.

    Returns:
        np.ndarray: Softmax-normalized tensor.
    """
    logits_max = np.max(input, axis=dim, keepdims=True)
    logits_stable = input - logits_max
    logits_exp = np.exp(logits_stable)
    logits_sum = np.sum(logits_exp, axis=dim, keepdims=True)
    return (logits_exp / logits_sum).astype(input.dtype, copy=False)


def tanh(input: np.ndarray) -> np.ndarray:
    """Applies the hyperbolic tangent activation function element-wise.

    Args:
        input (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Tensor with tanh applied.
    """
    return np.tanh(input)


def mse_loss(input: np.ndarray, target: np.ndarray) -> float:
    """Computes the Mean Squared Error (MSE) loss.

    Args:
        input (np.ndarray): Predicted values.
        target (np.ndarray): Ground truth values.

    Returns:
        float: Mean squared error value.
    """
    N = input.shape[0]
    return (np.mean((input - target) ** 2) / N).astype(np.float32)


def l1_loss(input: np.ndarray, target: np.ndarray) -> float:
    """Computes the L1 loss (Mean Absolute Error).

    Args:
        input (np.ndarray): Predicted values.
        target (np.ndarray): Ground truth values.

    Returns:
        float: Mean absolute error value.
    """
    return np.mean(np.abs(input - target), dtype=np.float32)


def cross_entropy(
    input: np.ndarray, target: np.ndarray, *, dim: int = 0, eps: float = 1e-8
) -> float:
    """Computes the cross-entropy loss for multi-class classification.

    Args:
        input (np.ndarray): Raw logits (before softmax).
        target (np.ndarray): Target class indices or one-hot encoded tensor.
        dim (int): Axis corresponding to class dimension. Default: 0.
        eps (float): Small epsilon for numerical stability. Default: 1e-8.

    Returns:
        float: Cross-entropy loss.
    """
    N = input.shape[0]
    y_hat = softmax(input, dim=dim)  # probabilities

    if target.ndim == 1:
        C = input.shape[1]
        target = np.eye(C, dtype=np.int32)[target]
    return -np.sum(target * np.log(y_hat + eps)) / N


def binary_cross_entropy(probs: np.ndarray, target: np.ndarray, eps: float) -> float:
    """Computes the binary cross-entropy loss.

    Args:
        probs (np.ndarray): Predicted probabilities.
        target (np.ndarray): Ground truth binary labels.
        eps (float): Small epsilon to prevent log(0).

    Returns:
        float: Binary cross-entropy loss.
    """
    return -np.mean(
        target * np.log(probs + eps) + (1 - target) * np.log(1 - probs + eps)
    )


def linear(
    input: np.ndarray, weight: np.ndarray, bias: Optional[Parameters] = None
) -> np.ndarray:
    """Applies a linear transformation: y = x W^T + b.

    Args:
        input (np.ndarray): Input tensor of shape (N, D_in).
        weight (np.ndarray): Weight matrix (D_out, D_in).
        bias (Optional[Parameters]): Bias term. Default: None.

    Returns:
        np.ndarray: Output tensor (N, D_out).
    """
    out = input @ weight.data.T  # Matrix multiplication
    if bias is not None:
        out += bias.data  # Add bias term
    return out


def flatten(input: np.ndarray) -> np.ndarray:
    """Flattens all dimensions except the batch dimension.

    Args:
        input (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Flattened tensor of shape (N, -1).
    """
    N = input.shape[0]
    return input.reshape(N, -1)


def conv1d(
    input: np.ndarray,
    weight: Parameters,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    *,
    bias: Optional[Parameters] = None,
    padding_mode: str = "zeros",
    extras: bool = False,
) -> ArrayAndExtras:
    """Performs a 1D convolution using an im2col approach with as_strided.

    Args:
        input (np.ndarray): Input tensor (N, C_in, L).
        weight (Parameters): Convolution weights (C_out, C_in, K).
        kernel_size (KernelSize): Size of the convolution kernel.
        stride (Stride): Stride value. Default: 1.
        padding (Padding): Padding value. Default: 0.
        bias (Optional[Parameters]): Bias term. Default: None.
        padding_mode (str): Padding type ('zeros', 'reflect', 'replicate', 'circular').
        extras (bool): If True, returns intermediate tensors.

    Returns:
        ArrayAndExtras: Output tensor or (out, col, w_col, Lout, x_p_shape) if extras=True.
    """

    def _add_padding(input: np.ndarray) -> np.ndarray:
        """Applies padding to the input tensor."""
        pad_width = ((0, 0), (0, 0), (padding, padding))
        padding_modes = ("zeros", "reflect", "replicate", "circular")
        if padding_mode in padding_modes:
            mode = (
                "constant"
                if padding_mode == "zeros"
                else (
                    "reflect"
                    if padding_mode == "reflect"
                    else "edge" if padding_mode == "replicate" else "wrap"
                )
            )
        else:
            raise ValueError(
                f"padding_mode Only accept {padding_modes} not '{padding_mode}'"
            )

        return np.pad(array=input, pad_width=pad_width, mode=mode)

    def _calc_out_size(input_seq: int) -> int:
        """Calculates the output sequence length."""
        return (input_seq + 2 * padding - kernel_size) // stride + 1

    def _im2col(input: np.ndarray, input_shape: Shape) -> tuple[np.ndarray, int]:
        """Performs im2col transformation for 1D convolution.

        Args:
            input: Input tensor (N, C, L).
            input_shape: Original shape of x.

        Returns:
            Tuple: (col matrix, output length L_out).
        """
        N, C, L = input_shape
        x_p = _add_padding(input)
        Lout = _calc_out_size(L)

        # Sliding windows
        shape = (N, C, Lout, kernel_size)
        sN, sC, sL = x_p.strides
        strides = (sN, sC, sL * stride, sL)
        windows = as_strided(x_p, shape=shape, strides=strides)
        col = windows.transpose(1, 3, 0, 2).reshape(C * kernel_size, -1)

        if extras:
            return col, Lout, x_p.shape
        return col, Lout

    out_channels = weight.data.shape[0]
    N = input.shape[0]

    if extras:
        col, Lout, x_p_shape = _im2col(input, input.shape)
    else:
        col, Lout = _im2col(input, input.shape)

    w_col = weight.data.reshape(out_channels, -1)
    out = w_col @ col
    if bias is not None:
        out += bias.data.reshape(out_channels, 1)

    out = out.reshape(out_channels, N, Lout).transpose(1, 0, 2)
    if extras:
        return out, col, w_col, Lout, x_p_shape
    return out


def conv2d(
    input: np.ndarray,
    weight: Parameters,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    *,
    bias: Optional[Parameters] = None,
    padding_mode: str = "zeros",
    extras: bool = False,
) -> ArrayAndExtras:
    """Performs a 2D convolution using im2col and as_strided.

    Args:
        input (np.ndarray): Input tensor (N, C_in, H, W).
        weight (Parameters): Convolution weights (C_out, C_in, KH, KW).
        kernel_size (KernelSize): Kernel dimensions.
        stride (Stride): Stride value. Default: 1.
        padding (Padding): Padding applied to input. Default: 0.
        bias (Optional[Parameters]): Bias term. Default: None.
        padding_mode (str): Padding type ('zeros', 'reflect', etc.).
        extras (bool): If True, returns intermediate matrices.

    Returns:
        ArrayAndExtras: Output tensor or (out, col, w_col) if extras=True.
    """

    def _pair(input: Padding | KernelSize | Stride) -> Tuple[int, int]:
        """Ensures input is a tuple pair (H, W)."""
        if isinstance(input, int):
            return (input, input)
        if isinstance(input, str):
            if input == "valid":
                return (0, 0)
            if input == "same":
                raise ValueError(f"The 'same' value is not currently supported")
            else:
                raise ValueError(f"Unsupported value '{input}'")
        return tuple(input)

    KH, KW = _pair(kernel_size)
    ph, pw = _pair(padding)
    sh, sw = _pair(stride)

    def _calc_out_size(height: int, width: int) -> Tuple[int, int]:
        """Computes output spatial dimensions."""
        out_h = (height + 2 * ph - KH) // sh + 1
        out_w = (width + 2 * pw - KW) // sw + 1
        return out_h, out_w

    def _add_padding(input: np.ndarray) -> np.ndarray:
        """Applies padding to the input tensor."""
        pad_width = ((0, 0), (0, 0), (ph, ph), (pw, pw))
        padding_modes = ("zeros", "reflect", "replicate", "circular")
        if padding_mode in padding_modes:
            mode = (
                "constant"
                if padding_mode == "zeros"
                else (
                    "reflect"
                    if padding_mode == "reflect"
                    else "edge" if padding_mode == "replicate" else "wrap"
                )
            )
        else:
            raise ValueError(
                f"padding_mode Only accept {padding_modes} not '{padding_mode}'"
            )

        return np.pad(array=input, pad_width=pad_width, mode=mode)

    def _im2col(input: np.ndarray, input_shape: Shape) -> tuple[np.ndarray, int, int]:
        """Performs im2col transformation for 2D convolution.
        Args:
            input: Input tensor (N, C, H, W).
            input_shape: Original shape of x.

        Returns:
            Tuple: (col matrix, output height, output width).
        """
        N, C, H, W = input_shape
        x_padded = _add_padding(input)

        out_height, out_width = _calc_out_size(H, W)

        # Use as_strided to create sliding windows
        shape = (N, C, out_height, out_width, KH, KW)
        sN, sC, sH, sW = x_padded.strides
        strides = (sN, sC, sH * sh, sW * sw, sH, sW)
        windows = as_strided(x=x_padded, shape=shape, strides=strides)

        # Reshape to column matrix (C*KH*KW, N*out_h*out_w)
        col = windows.transpose(1, 4, 5, 0, 2, 3).reshape(C * KH * KW, -1)
        return col, out_height, out_width

    out_channels = weight.data.shape[0]
    N = input.shape[0]

    col, out_height, out_width = _im2col(input, input.shape)
    w_col = weight.data.reshape(out_channels, -1)

    # Convolution as Matrix Multiplication: (out_c, C*K*K) @ (C*K*K, N*out_h*out_w) -> (out_c, N*out_h*out_w)
    out = w_col @ col
    if bias is not None:
        out += bias.data.reshape(out_channels, 1)

    # Reshape back to (N, C_out, H_out, W_out)
    out = out.reshape(out_channels, N, out_height, out_width).transpose(1, 0, 2, 3)
    if extras:
        return out, col, w_col
    return out


def max_pool1d(
    input: np.ndarray,
    kernel_size: KernelSize,
    stride: Stride,
    padding: Padding = 0,
    padding_mode: str = "zeros",
    extras: bool = False,
) -> ArrayAndExtras:
    """Applies 1D max pooling operation.

    Args:
        input (np.ndarray): Input tensor (N, C, L).
        kernel_size (KernelSize): Size of pooling window.
        stride (Stride): Step size between windows.
        padding (Padding): Optional padding. Default: 0.
        padding_mode (str): Type of padding. Default: 'zeros'.
        extras (bool): If True, returns extra window data.

    Returns:
        ArrayAndExtras: Pooled output or (out, windows, Lout, x_p_shape).
    """

    def _add_padding(input: np.ndarray) -> np.ndarray:
        """Applies padding."""
        pad_width = ((0, 0), (0, 0), (padding, padding))
        mode = (
            "constant"
            if padding_mode == "zeros"
            else (
                "reflect"
                if padding_mode == "reflect"
                else "edge" if padding_mode == "replicate" else "wrap"
            )
        )
        return np.pad(array=input, pad_width=pad_width, mode=mode)

    def _calc_out_length(L: int) -> int:
        """Computes pooled sequence length."""
        return (L + 2 * padding - kernel_size) // stride + 1

    def _get_windows(input: np.ndarray) -> Tuple[np.ndarray, int, tuple]:
        """Creates sliding windows using as_strided."""
        N, C, L = input.shape
        x_p = _add_padding(input)
        L_out = _calc_out_length(L)

        # Shape of the windows: (N, C, L_out, K)
        shape = (N, C, L_out, kernel_size)
        sN, sC, sL = x_p.strides
        strides = (sN, sC, sL * stride, sL)

        windows = as_strided(x_p, shape=shape, strides=strides)

        return windows, L_out, x_p.shape

    windows, Lout, x_p_shape = _get_windows(input)
    out = windows.max(axis=3)  # Shape (N, C, L_out)
    if extras:
        return out, windows, Lout, x_p_shape
    return out


def max_pool2d(
    input: np.ndarray,
    kernel_size: KernelSize,
    stride: Stride,
    padding: Padding = 0,
    padding_mode: str = "zeros",
    extras: bool = False,
) -> ArrayAndExtras:
    """Applies 2D max pooling operation.

    Args:
        input (np.ndarray): Input tensor (N, C, H, W).
        kernel_size (KernelSize): Pooling window size.
        stride (Stride): Stride between windows.
        padding (Padding): Optional padding. Default: 0.
        padding_mode (str): Padding type. Default: 'zeros'.
        extras (bool): If True, returns window details.

    Returns:
        ArrayAndExtras: Pooled tensor or (out, windows, Hout, Wout, x_p_shape).
    """

    def _pair(input: Padding | KernelSize | Stride) -> Tuple[int, int]:
        """Converts integer, tuple, or valid/same string to a (H, W) pair."""
        if isinstance(input, int):
            return (input, input)
        if isinstance(input, str):
            if input == "valid":
                return (0, 0)
            if input == "same":
                raise ValueError(f"The 'same' value is not currently supported")
            else:
                raise ValueError(f"Unsupported value '{input}'")
        return tuple(input)

    KH, KW = _pair(kernel_size)
    ph, pw = _pair(padding)
    sh, sw = _pair(stride)

    def _calc_out_size(height: int, width: int) -> Tuple[int, int]:
        """Calculates the output height and width (H_out, W_out)."""
        out_height = (height + 2 * ph - KH) // sh + 1
        out_width = (width + 2 * pw - KW) // sw + 1
        return out_height, out_width

    def _add_padding(input: np.ndarray) -> np.ndarray:
        """Applies padding to the input tensor."""
        pad_width = ((0, 0), (0, 0), (ph, ph), (pw, pw))
        padding_modes = ("zeros", "reflect", "replicate", "circular")
        if padding_mode in padding_modes:
            mode = (
                "constant"
                if padding_mode == "zeros"
                else (
                    "reflect"
                    if padding_mode == "reflect"
                    else "edge" if padding_mode == "replicate" else "wrap"
                )
            )
        else:
            raise ValueError(
                f"padding_mode Only accept {padding_modes} not '{padding_mode}'"
            )

        return np.pad(array=input, pad_width=pad_width, mode=mode)

    def _get_windows(input: np.ndarray) -> Tuple[np.ndarray, int, int, tuple]:
        """Creates sliding windows using as_strided."""
        N, C, H, W = input.shape
        x_p = _add_padding(input)
        out_height, out_width = _calc_out_size(H, W)

        # Shape of the windows: (N, C, out_h, out_w, KH, KW)
        shape = (N, C, out_height, out_width, KH, KW)
        sN, sC, sH, sW = x_p.strides
        strides = (sN, sC, sH * sh, sW * sw, sH, sW)

        windows = as_strided(x=x_p, shape=shape, strides=strides)

        return windows, out_height, out_width, x_p.shape

    windows, out_height, out_width, x_p_shape = _get_windows(input)
    out = windows.max(axis=(4, 5))  # Shape (N, C, out_h, out_w)
    if extras:
        return out, windows, out_height, out_width, x_p_shape
    return out


def avg_pool1d(input: np.ndarray) -> np.ndarray:
    """Applies 1D average pooling over length dimension.

    Args:
        input (np.ndarray): Input tensor (N, C, L).

    Returns:
        np.ndarray: Averaged tensor (N, C, 1).
    """
    return np.mean(input, axis=2, keepdims=True)


def avg_pool2d(input: np.ndarray) -> np.ndarray:
    """Applies 2D average pooling over spatial dimensions.

    Args:
        input (np.ndarray): Input tensor (N, C, H, W).

    Returns:
        np.ndarray: Averaged tensor (N, C, 1, 1).
    """
    return np.mean(input, axis=(2, 3), keepdims=True)
