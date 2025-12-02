import numpy as np
from typing import Optional

from novann.module.layer import Layer


class Dropout(Layer):
    """Dropout regularization layer.

    During training, randomly zeros elements of the input with probability `p`
    (drop probability) and scales the remaining elements by 1 / (1 - p) so that
    expected activations are preserved. In evaluation mode the layer is a no-op.

    Args:
        p: Drop probability in [0, 1). p == 0 means no dropout. p must be < 1.0.

    Note:
        - The layer clears its internal mask after backward to avoid keeping
          references between batches.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("Dropout probability must be in the interval [0.0, 1.0).")
        self.p: float = float(p)
        self._mask: Optional[np.ndarray] = None

    def train(self) -> None:
        """Set layer to training mode and clear any existing mask."""
        super().train()
        self._mask = None

    def eval(self) -> None:
        """Set layer to evaluation mode and clear any existing mask."""
        super().eval()
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout in training, or return input unchanged in eval.

        Args:
            x: Input array.

        Returns:
            Array after applying dropout (training) or same input (eval).
        """
        if not self._training or self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p
        # Create mask of same dtype as input (1.0 = keep, 0.0 = drop)
        self._mask = (np.random.rand(*x.shape) < keep_prob).astype(x.dtype)
        return (x * self._mask) / keep_prob

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backprop through dropout: apply the same mask and scaling.

        Args:
            grad: Gradient w.r.t. the output.

        Returns:
            Gradient w.r.t. the input.
        """
        if not self._training:
            return grad

        keep_prob = 1.0 - self.p
        res = (grad * self._mask) / keep_prob
        self._mask = None
        return res

    def __repr__(self):
        return f"Dropout(p={self.p})"
