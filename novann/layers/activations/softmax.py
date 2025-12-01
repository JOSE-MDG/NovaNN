import numpy as np
from typing import Optional

from novann.layers.activations.activations import Activation


class SoftMax(Activation):
    """SoftMax activation layer.

    Computes a numerically stable softmax along a specified axis. The layer
    does not affect weight initialization (affect_init=False).

    Attributes:
        axis (int): Axis along which to apply softmax.
        out (Optional[np.ndarray]): Cached output from the forward pass.
    """

    def __init__(self, axis: int = 1) -> None:
        """Initialize SoftMax.

        Args:
            axis: Axis to normalize over (default: 1).
        """
        super().__init__()
        self.affect_init = False
        self.out: Optional[np.ndarray] = None
        self.axis: int = int(axis)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: compute numerically stable softmax.

        Args:
            x: Input logits array.

        Returns:
            Softmax probabilities with same shape as `x`.
        """
        logits_max = np.max(x, axis=self.axis, keepdims=True)
        stable_logits = x - logits_max
        logits_exp = np.exp(stable_logits)
        logits_sum = np.sum(logits_exp, axis=self.axis, keepdims=True)
        self.out = logits_exp / logits_sum
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

        s = np.sum(self.out * grad, axis=self.axis, keepdims=True)
        res = self.out * (grad - s)
        return res
