import numpy as np
import novann.functional as F

from typing import Optional
from novann.layers.activations import Activation


class Softmax(Activation):
    """SoftMax activation layer.

    Computes a numerically stable softmax along a specified axis. The layer
    does not affect weight initialization (affect_init=False).

    Attributes:
        axis (int): Axis along which to apply softmax.
        out (Optional[np.ndarray]): Cached output from the forward pass.
    """

    def __init__(self, dim: int = 1) -> None:
        """Initialize SoftMax.

        Args:
            axis: Axis to normalize over (default: 1).
        """
        super().__init__()
        self.affect_init = False
        self.out: Optional[np.ndarray] = None
        self.dim: int = int(dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: compute numerically stable softmax.

        Args:
            x: Input logits array.

        Returns:
            Softmax probabilities with same shape as `x`.
        """
        self.out = F.softmax(x, dim=self.dim)
        return self.out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass: compute gradient w.r.t. input using cached output.

        Uses the Jacobian-vector product for softmax:
            J_softmax * grad = softmax * (grad - sum(softmax * grad, axis))

        Args:
            grad: Gradient w.r.t. the output of this layer.

        Returns:
            Gradient w.r.t. the input of this layer.
        """

        s = np.sum(self.out * grad, axis=self.dim, keepdims=True)
        res = self.out * (grad - s)
        return res

    def __repr__(self):
        return f"SoftMax(dim={self.dim})"
