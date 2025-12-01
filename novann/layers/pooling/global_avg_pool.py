import numpy as np
from novann.module import Layer


class GlobalAvgPool1d(Layer):
    """Applies Global Average Pooling over the length dimension (L) of a 1D input.

    Input shape: (N, C, L_in)
    Output shape: (N, C, 1)

    This layer has no trainable parameters.
    """

    def __init__(self):
        """Initialize the layer."""
        super().__init__()
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 1D Global Average Pooling.

        Args:
            x (np.ndarray): Input array with shape (N, C, L).

        Returns:
            np.ndarray: Output array with shape (N, C, 1).
        """
        x = x.astype(np.float32, copy=False)  # (N,C,L)

        # Calculates the mean over the length dimension (axis=2)
        output = x.mean(axis=2, keepdims=True)
        self._cache["x_shape"] = x.shape
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for 1D Global Average Pooling.

        The gradient is distributed uniformly back to the input, divided by the
        length of the original input.

        Args:
            grad_output (np.ndarray): Gradient w.r.t. the layer output,
                                      shape (N, C, 1).

        Returns:
            np.ndarray: Gradient w.r.t. the layer input,
                        shape (N, C, L).
        """
        grad_output = grad_output.astype(np.float32, copy=False)
        N, C, L = self._cache["x_shape"]

        # Distribution factor: 1 / L
        factor = 1.0 / L

        # Tile the gradient across the length dimension
        grad_input = np.ones((N, C, L), dtype=np.float32) * (grad_output * factor)
        return grad_input


class GlobalAvgPool2d(Layer):
    """Applies Global Average Pooling over the spatial dimensions (H, W) of a 2D input.

    Input shape: (N, C, H_in, W_in)
    Output shape: (N, C, 1, 1)

    This layer has no trainable parameters.
    """

    def __init__(self):
        """Initialize the layer."""
        super().__init__()
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 2D Global Average Pooling.

        Args:
            x (np.ndarray): Input array with shape (N, C, H, W).

        Returns:
            np.ndarray: Output array with shape (N, C, 1, 1).
        """
        x = x.astype(np.float32, copy=False)

        # Calculates the mean over the spatial dimensions (axis=2, 3)
        output = x.mean(axis=(2, 3), keepdims=True)
        self._cache["x_shape"] = x.shape
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for 2D Global Average Pooling.

        The gradient is distributed uniformly back to the input, divided by the
        area (H * W) of the original input.

        Args:
            grad_output (np.ndarray): Gradient w.r.t. the layer output,
                                      shape (N, C, 1, 1).

        Returns:
            np.ndarray: Gradient w.r.t. the layer input,
                        shape (N, C, H, W).
        """
        grad_output = grad_output.astype(np.float32, copy=False)
        N, C, H, W = self._cache["x_shape"]

        # Distribution factor: 1 / (H * W)
        factor = 1.0 / (H * W)

        # Tile the gradient across the spatial dimensions
        grad_input = np.ones((N, C, H, W), dtype=np.float32) * (grad_output * factor)
        return grad_input
