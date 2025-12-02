import numpy as np
from typing import Optional

from novann.layers.activations.activations import Activation


class Tanh(Activation):
    """Hyperbolic tangent activation layer.

    Caches the forward output to compute the derivative during backward pass.

    Attributes:
        out (Optional[np.ndarray]): Cached output from the forward pass.
    """

    def __init__(self) -> None:
        """Initialize Tanh activation."""
        super().__init__()
        self.affect_init: bool = True
        self.out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute forward pass.

        Args:
            x: Input array.

        Returns:
            The tanh applied element-wise.
        """
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Compute backward pass.

        Uses cached forward output:
            d/dx tanh(x) = 1 - tanh(x)^2

        Args:
            grad: Gradient w.r.t. the output of this layer.

        Returns:
            Gradient w.r.t. the input of this layer.
        """

        res = grad * (1.0 - self.out**2)
        return res

    def __repr__(self):
        return "Tanh()"
