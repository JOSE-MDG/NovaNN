import numpy as np
from numpy.lib.stride_tricks import as_strided
from novann._typing import IntOrPair
from typing import Optional, Tuple
from novann.module import Layer


class MaxPool1d(Layer):
    """Applies a 1D max pooling over an input signal.

    The input must have shape (N, C, L_in).

    Args:
        kernel_size (int): Size of the pooling window.
        stride (int | None): Stride of the pooling. If None, defaults to `kernel_size`.
        padding (int): Zero-padding added to both sides of the input. Default: 0.
        padding_mode (str): Padding mode ('zeros', 'reflect', 'replicate', 'circular'). Default: 'zeros'.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.K: int = int(kernel_size)
        # Default stride is equal to kernel size (non-overlapping)
        self.stride: int = int(stride) if stride is not None else self.K
        self.padding: int = int(padding)
        self.pm: str = padding_mode
        self._cache: dict = {}

    def _add_padding(self, x: np.ndarray) -> np.ndarray:
        """Applies padding to the input tensor."""
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding))
        padding_modes = ("zeros", "reflect", "replicate", "circular")
        if self.pm in padding_modes:
            if self.pm == "zeros":
                mode = "constant"
            elif self.pm == "reflect":
                mode = "reflect"
            elif self.pm == "replicate":
                mode = "edge"
            else:
                mode = "wrap"
        else:
            raise ValueError(
                f"padding_mode Only accept {padding_modes} not '{self.pm}'"
            )

        return np.pad(array=x, pad_width=pad_width, mode=mode)

    def _calc_out_length(self, L: int) -> int:
        """Calculates the output length L_out."""
        L_out = (L + 2 * self.padding - self.K) // self.stride + 1
        return L_out

    def _get_windows(self, x: np.ndarray) -> Tuple[np.ndarray, int, tuple]:
        """Creates sliding windows using as_strided."""
        N, C, L = x.shape
        x_p = self._add_padding(x)
        L_out = self._calc_out_length(L)

        # Shape of the windows: (N, C, L_out, K)
        shape = (N, C, L_out, self.K)
        sN, sC, sL = x_p.strides
        strides = (sN, sC, sL * self.stride, sL)

        windows = as_strided(x_p, shape=shape, strides=strides)

        return windows, L_out, x_p.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 1D Max Pooling.

        Args:
            x (np.ndarray): Input array with shape (N, C, L_in).

        Returns:
            np.ndarray: Output array with shape (N, C, L_out).
        """
        x = x.astype(np.float32, copy=False)
        windows, L_out, padded_shape = self._get_windows(x)

        # Max operation over the kernel length (axis=3)
        out = windows.max(axis=3)  # Shape (N, C, L_out)

        self._cache["x_shape"] = x.shape
        self._cache["windows"] = windows
        self._cache["L_out"] = L_out
        self._cache["padded_shape"] = padded_shape
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for 1D Max Pooling (unpooling).

        Args:
            grad_output (np.ndarray): Gradient w.r.t. the layer output,
                                      shape (N, C, L_out).

        Returns:
            np.ndarray: Gradient w.r.t. the layer input,
                        shape (N, C, L_in).
        """
        grad_output = grad_output.astype(np.float32, copy=False)
        windows = self._cache["windows"]
        x_shape = self._cache["x_shape"]
        L_out = self._cache["L_out"]
        padded_shape = self._cache["padded_shape"]
        N, C, L = x_shape

        # 1. Create a mask: find the max values within each window
        # max_vals shape: (N, C, L_out, 1)
        max_vals = windows.max(axis=3, keepdims=True)
        mask = (windows == max_vals).astype(np.float32)

        # Handle multiple max elements (split the gradient)
        counts = mask.sum(axis=3, keepdims=True)
        counts[counts == 0] = 1.0  # Avoid division by zero
        grad_output_expand = grad_output[:, :, :, np.newaxis]  # (N, C, L_out, 1)

        # 2. Distribute gradient to max locations within the windows
        grad_windows = mask * (grad_output_expand / counts)

        # 3. Scatter (Col2Im equivalent): sum gradients back to the input
        grad_input_padded = np.zeros(padded_shape, dtype=np.float32)

        # NOTE: Potential performance bottleneck (Python loop)
        for l in range(L_out):
            start = l * self.stride
            grad_input_padded[:, :, start : start + self.K] += grad_windows[:, :, l, :]

        self._cache.clear()

        if self.padding == 0:
            return grad_input_padded
        # Remove padding
        return grad_input_padded[:, :, self.padding : self.padding + L]


class MaxPool2d(Layer):
    """Applies a 2D max pooling over an input signal.

    The input must have shape (N, C, H_in, W_in).

    Args:
        kernel_size (IntOrPair): Size of the pooling window (H, W).
        stride (IntOrPair | None): Stride of the pooling (H, W). If None, defaults to `kernel_size`.
        padding (IntOrPair): Zero-padding added to both sides (H, W). Default: 0.
        padding_mode (str): Padding mode ('zeros', 'reflect', 'replicate', 'circular'). Default: 'zeros'.
    """

    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: Optional[IntOrPair] = None,
        padding: IntOrPair = 0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.KH, self.KW = self._pair(kernel_size)
        # Default stride is equal to kernel size (non-overlapping)
        stride_val = stride if stride is not None else kernel_size
        self.sh, self.sw = self._pair(stride_val)
        self.ph, self.pw = self._pair(padding)
        self.pm: str = padding_mode
        self._cache: dict = {}

    def _pair(self, x: IntOrPair) -> Tuple[int, int]:
        """Converts integer or tuple to a (H, W) pair."""
        if isinstance(x, int):
            return (x, x)
        return tuple(x)

    def _add_padding(self, x: np.ndarray) -> np.ndarray:
        """Applies padding to the input tensor."""
        pad_width = ((0, 0), (0, 0), (self.ph, self.ph), (self.pw, self.pw))
        padding_modes = ("zeros", "reflect", "replicate", "circular")
        if self.pm in padding_modes:
            if self.pm == "zeros":
                mode = "constant"
            elif self.pm == "reflect":
                mode = "reflect"
            elif self.pm == "replicate":
                mode = "edge"
            else:
                mode = "wrap"
        else:
            raise ValueError(
                f"padding_mode Only accept {padding_modes} not '{self.pm}'"
            )

        return np.pad(array=x, pad_width=pad_width, mode=mode)

    def _calc_out_size(self, H: int, W: int) -> Tuple[int, int]:
        """Calculates the output height and width (H_out, W_out)."""
        out_height = (H + 2 * self.ph - self.KH) // self.sh + 1
        out_width = (W + 2 * self.pw - self.KW) // self.sw + 1
        return out_height, out_width

    def _get_windows(self, x: np.ndarray) -> Tuple[np.ndarray, int, int, tuple]:
        """Creates sliding windows using as_strided."""
        N, C, H, W = x.shape
        x_p = self._add_padding(x)
        out_height, out_width = self._calc_out_size(H, W)

        # Shape of the windows: (N, C, out_h, out_w, KH, KW)
        shape = (N, C, out_height, out_width, self.KH, self.KW)
        sN, sC, sH, sW = x_p.strides
        strides = (sN, sC, sH * self.sh, sW * self.sw, sH, sW)

        windows = as_strided(x=x_p, shape=shape, strides=strides)

        return windows, out_height, out_width, x_p.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 2D Max Pooling.

        Args:
            x (np.ndarray): Input array with shape (N, C, H_in, W_in).

        Returns:
            np.ndarray: Output array with shape (N, C, H_out, W_out).
        """
        x = x.astype(np.float32, copy=False)
        windows, out_h, out_w, padded_shape = self._get_windows(x)

        # Max operation over the kernel dimensions (axis 4 and 5)
        out = windows.max(axis=(4, 5))  # Shape (N, C, out_h, out_w)

        self._cache["x_shape"] = x.shape
        self._cache["windows"] = windows
        self._cache["padded_shape"] = padded_shape
        self._cache["out_h"] = out_h
        self._cache["out_w"] = out_w
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for 2D Max Pooling (unpooling).

        Args:
            grad_output (np.ndarray): Gradient w.r.t. the layer output,
                                      shape (N, C, H_out, W_out).

        Returns:
            np.ndarray: Gradient w.r.t. the layer input,
                        shape (N, C, H_in, W_in).
        """
        grad_output = grad_output.astype(np.float32, copy=False)
        windows = self._cache["windows"]
        x_shape = self._cache["x_shape"]
        padded_shape = self._cache["padded_shape"]
        out_h = self._cache["out_h"]
        out_w = self._cache["out_w"]
        _, _, H, W = x_shape

        # 1. Create a mask: find the max values within each window
        # max_vals shape: (N, C, out_h, out_w, 1, 1)
        max_vals = windows.max(axis=(4, 5), keepdims=True)
        mask = (windows == max_vals).astype(np.float32)

        # Handle multiple max elements (split the gradient)
        counts = mask.sum(axis=(4, 5), keepdims=True)
        counts[counts == 0] = 1.0
        grad_output_expand = grad_output[:, :, :, :, np.newaxis, np.newaxis]

        # 2. Distribute gradient to max locations within the windows
        grad_windows = mask * (grad_output_expand / counts)
        grad_input_padded = np.zeros(padded_shape, dtype=np.float32)

        # 3. Scatter (Col2Im equivalent): sum gradients back to the input
        # NOTE: Logical bug fix applied in index calculation below:
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.sh
                h_end = h_start + self.KH  # Corrected from self.sh
                w_start = j * self.sw  # Corrected from j + self.sw
                w_end = w_start + self.KW  # Corrected from self.sw

                grad_input_padded[:, :, h_start:h_end, w_start:w_end] += grad_windows[
                    :, :, i, j, :, :
                ]

        self._cache.clear()

        if self.ph == 0 and self.pw == 0:
            return grad_input_padded
        # Remove padding
        return grad_input_padded[:, :, self.ph : self.ph + H, self.pw : self.pw + W]
