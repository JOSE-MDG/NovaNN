import numpy as np
from typing import Optional

from src.layers.activations.activations import Activation


class Sigmoid(Activation):
    """Sigmoid activation layer.

    Applies the element-wise sigmoid function:
        sigma(x) = 1 / (1 + exp(-x))

    This layer stores the last forward output to compute the gradient during
    the backward pass.

    Attributes:
        out (Optional[np.ndarray]): Cached output from the forward pass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.affect_init: bool = True
        self.out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the sigmoid.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid applied element-wise.
        """
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass computing gradient w.r.t. input.

        Uses cached forward output to compute derivative:
            d/dx sigma(x) = sigma(x) * (1 - sigma(x))

        Args:
            grad (np.ndarray): Gradient w.r.t. the output of this layer.

        Returns:
            np.ndarray: Gradient w.r.t. the input of this layer.
        """

        res = grad * (self.out * (1.0 - self.out))
        return res
