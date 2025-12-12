import numpy as np
import novann.functional as F
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 1D Max Pooling.

        Args:
            x (np.ndarray): Input array with shape (N, C, L_in).

        Returns:
            np.ndarray: Output array with shape (N, C, L_out).
        """
        x = x.astype(np.float32, copy=False)

        out, windows, L_out, padded_shape = F.max_pool1d(
            x, self.K, self.stride, self.padding, extras=True
        )
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

        # max_vals shape: (N, C, L_out, 1)
        max_vals = windows.max(axis=3, keepdims=True)
        mask = (windows == max_vals).astype(np.float32)

        # Handle multiple max elements (split the gradient)
        counts = mask.sum(axis=3, keepdims=True)
        counts[counts == 0] = 1.0  # Avoid division by zero
        grad_output_expand = grad_output[:, :, :, np.newaxis]  # (N, C, L_out, 1)

        # Distribute gradient to max locations within the windows
        grad_windows = mask * (grad_output_expand / counts)

        # scatter: sum gradients back to the input
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

    def __repr__(self):
        return f"MaxPool1d(kernel_size={self.K}, stride={self.stride}, padding={self.padding})"
