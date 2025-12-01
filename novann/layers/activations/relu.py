import numpy as np
from typing import Optional

from novann.layers.activations.activations import Activation


class ReLU(Activation):
    """Rectified Linear Unit (ReLU) activation layer.

    This layer applies element-wise max(0, x). It stores a boolean mask
    during the forward pass to compute gradients efficiently in backward.

    Attributes:
        _mask: Boolean mask saved from the forward pass.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mask: Optional[np.ndarray] = None
        self.affect_init = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute forward pass.

        Args:
            x: Input array.

        Returns:
            Array with ReLU applied element-wise.
        """
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Compute backward pass.

        Args:
            grad: Gradient w.r.t. the output of this layer.

        Returns:
            Gradient w.r.t. the input of this layer.
        """
        return grad * self._mask


class LeakyReLU(Activation):
    """Leaky ReLU activation layer.

    Provides a small slope for negative inputs to avoid dead neurons.

    Attributes:
        a: Negative slope coefficient.
        activation_param: same value as `a`
        _cache_input: Saved input for backward computation.
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.a: float = negative_slope
        self.affect_init = True
        self.activation_param = negative_slope
        self._cache_input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute forward pass.

        Args:
            x: Input array.

        Returns:
            Array with LeakyReLU applied element-wise.
        """
        self._cache_input = x
        return np.where(x >= 0, x, self.a * x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Compute backward pass.

        Args:
            grad: Gradient w.r.t. the output of this layer.

        Returns:
            Gradient w.r.t. the input of this layer.
        """
        x = self._cache_input
        return grad * np.where(x >= 0, 1.0, self.a)
