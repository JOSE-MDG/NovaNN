import numpy as np
import novann.functional as F
from novann._typing import Padding, KernelSize, Stride
from typing import Tuple
from novann.module import Layer


class MaxPool2d(Layer):
    """Applies a 2D max pooling over an input signal.

    The input must have shape (N, C, H_in, W_in).

    Args:
        kernel_size (KernelSize): Size of the pooling window (H, W).
        stride (Stride): Stride of the pooling (H, W). If None, defaults to `kernel_size`.
        padding (Padding): Zero-padding added to both sides (H, W). Default: 0.
        padding_mode (str): Padding mode ('zeros', 'reflect', 'replicate', 'circular'). Default: 'zeros'.
    """

    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Stride = None,
        padding: Padding = 0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.KH, self.KW = self._pair(kernel_size)

        stride_val = stride if stride is not None else kernel_size
        self.sh, self.sw = self._pair(stride_val)

        self.ph, self.pw = self._pair(padding)
        self.pm: str = padding_mode
        self.pm = padding_mode
        self._cache: dict = {}

    def _pair(self, x: Padding | KernelSize | Stride) -> Tuple[int, int]:
        """Converts integer, tuple, or valid/same string to a (H, W) pair."""
        if isinstance(x, int):
            return (x, x)
        if isinstance(x, str):
            if x == "valid":
                return (0, 0)
            if x == "same":
                raise ValueError(f"The 'same' value is not currently supported")
            else:
                raise ValueError(f"Unsupported value '{x}'")
        return tuple(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 2D Max Pooling.

        Args:
            x (np.ndarray): Input array with shape (N, C, H_in, W_in).

        Returns:
            np.ndarray: Output array with shape (N, C, H_out, W_out).
        """
        x = x.astype(np.float32, copy=False)

        # Max operation over the kernel dimensions (axis 4 and 5)
        out, windows, out_h, out_w, padded_shape = F.max_pool2d(
            x,
            (self.KH, self.KW),
            (self.sh, self.sw),
            (self.ph, self.pw),
            padding_mode=self.pm,
            extras=True,
        )

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
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.sh
                h_end = h_start + self.KH
                w_start = j * self.sw
                w_end = w_start + self.KW

                grad_input_padded[:, :, h_start:h_end, w_start:w_end] += grad_windows[
                    :, :, i, j, :, :
                ]

        self._cache.clear()

        if self.ph == 0 and self.pw == 0:
            return grad_input_padded
        # Remove padding
        return grad_input_padded[:, :, self.ph : self.ph + H, self.pw : self.pw + W]

    def __repr__(self):
        return f"MaxPool2d(kernel_size={(self.KH, self.KW)}, stride={(self.sh, self.sw)}, padding={(self.ph,self.pw)})"
