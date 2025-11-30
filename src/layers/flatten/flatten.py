import numpy as np
from src.module import Layer
from typing import Optional


class Flatten(Layer):
    """Reshapes the input tensor by flattening all dimensions except the batch dimension (axis 0).

    This layer is typically used to connect a convolutional/pooling layer output
    to a fully connected (Linear) layer input.

    It has no trainable parameters.
    """

    def __init__(self):
        """Initialize the layer."""
        super().__init__()
        # Store the original shape for the backward pass (unflattening)
        self._origin_shape: Optional[tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: flatten the input.

        Args:
            x (np.ndarray): Input tensor of shape (N, D1, D2, ...).

        Returns:
            np.ndarray: Flattened output tensor of shape (N, D1 * D2 * ...).
        """
        # Cache the input shape (N, D1, D2, ...)
        self._origin_shape = x.shape
        N = x.shape[0]

        # Reshape to (N, -1), where -1 is the product of all remaining dimensions
        return x.reshape(N, -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass: unflatten the gradient to the original input shape.

        Args:
            grad (np.ndarray): Gradient w.r.t. the layer output,
                               shape (N, Total_Features).

        Returns:
            np.ndarray: Gradient w.r.t. the layer input,
                        restored to shape (*_origin_shape).
        """
        # Restore the original shape, maintaining the gradient flow
        return grad.reshape(*self._origin_shape)
