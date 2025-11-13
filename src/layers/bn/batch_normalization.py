import numpy as np
from typing import Optional, List

from src.module.layer import Layer
from src.module.module import Parameters


class BatchNormalization(Layer):
    """Batch normalization layer for fully-connected inputs.

    Normalizes activations per feature using running estimates during eval
    and batch statistics during training. Implements learnable scale (gamma)
    and shift (beta) parameters.

    Args:
        num_features: Number of features (feature dimension).
        momentum: Running statistics momentum (default 0.1).
        eps: Small epsilon to stabilize division (default 1e-5).

    Attributes:
        gamma: Scale parameter wrapper (Parameters).
        beta: Shift parameter wrapper (Parameters).
        running_mean: Running mean (shape (1, num_features)).
        running_var: Running variance (shape (1, num_features)).
        x, x_hat, mu, var, x_mu, m: Cached intermediate values used in backward.
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features: int = int(num_features)
        self.momentum: float = float(momentum)
        self.eps: float = float(eps)

        self.gamma: Parameters = Parameters(np.ones((1, self.num_features)))
        self.beta: Parameters = Parameters(np.zeros((1, self.num_features)))
        self.beta.name = "beta"
        self.gamma.name = "gamma"

        self.running_mean: np.ndarray = np.zeros((1, self.num_features))
        self.running_var: np.ndarray = np.ones((1, self.num_features))

        # Caches used during forward/backward
        self.x: Optional[np.ndarray] = None
        self.x_hat: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self.x_mu: Optional[np.ndarray] = None
        self.m: Optional[int] = None

    def eval(self) -> None:
        """Set layer to evaluation mode and preserve running statistics."""
        super().eval()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for batch normalization.

        Uses batch statistics when training, otherwise uses running estimates.

        Args:
            x: Input array with shape (batch, features).

        Returns:
            Normalized, scaled and shifted output with same shape as x.
        """
        self.x = x
        self.m = x.shape[0]
        if self._training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            x_mu = x - mu
            x_hat = x_mu / np.sqrt(var + self.eps)

            self.mu = mu
            self.var = var
            self.x_mu = x_mu
            self.x_hat = x_hat

            out = self.gamma.data * x_hat + self.beta.data

            # Update running statistics
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mu
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var

            return out
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma.data * x_hat + self.beta.data
            return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients for parameters and inputs.

        Computes gradients for gamma and beta (stored in their .grad) and
        returns gradient with respect to the input.

        Args:
            grad: Gradient w.r.t. the output, shape (batch, features).

        Returns:
            Gradient with respect to the input, same shape as grad.
        """
        # Local aliases to clarify math
        m = self.m
        x_hat = self.x_hat
        var = self.var
        x_mu = self.x_mu
        eps = self.eps

        # Parameter gradients
        self.gamma.grad = np.sum(grad * x_hat, axis=0, keepdims=True)
        self.beta.grad = np.sum(grad, axis=0, keepdims=True)

        dx_hat = grad * self.gamma.data

        inv_std = (var + eps) ** (-0.5)
        inv_std3 = (var + eps) ** (-1.5)

        dvar = np.sum(dx_hat * x_mu * (-0.5) * inv_std3, axis=0, keepdims=True)
        dmu = (
            np.sum(dx_hat * (-inv_std), axis=0, keepdims=True)
            + dvar * np.sum(-2.0 * x_mu, axis=0, keepdims=True) / m
        )
        dx = dx_hat * inv_std + dvar * (2.0 * x_mu) / m + dmu / m
        return dx

    def parameters(self) -> List[Parameters]:
        """Return the list of Parameters objects owned by this layer."""
        return [self.gamma, self.beta]
