import numpy as np
from typing import Optional
from novann._typing import ListOfParameters

from novann.module import Layer, Parameters


class BatchNorm2d(Layer):
    """Batch Normalization layer for 2D convolutional inputs (4D tensors).

    Normalizes activations per channel across spatial dimensions and batch.
    Uses running estimates during evaluation and batch statistics during training.

    Args:
        num_features: Number of input channels.
        momentum: Momentum for running statistics (default: 0.1).
        eps: Epsilon for numerical stability (default: 1e-5).

    Attributes:
        gamma: Scale parameter (shape: [1, channels, 1, 1]).
        beta: Shift parameter (shape: [1, channels, 1, 1]).
        running_mean: Running mean estimate.
        running_var: Running variance estimate.
        x: Cached input from forward pass.
        x_hat: Normalized input.
        mu: Batch mean.
        var: Batch variance.
        x_mu: Input minus mean.
        m: Effective batch size (batch_size * height * width).
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        """Initialize parameters, running statistics, and cache."""
        super().__init__()
        self.num_features: int = int(num_features)
        self.momentum: float = float(momentum)
        self.eps: float = float(eps)

        # Parameters with 4D shape for convolutional inputs
        self.gamma: Parameters = Parameters(np.ones((1, self.num_features, 1, 1)))
        self.beta: Parameters = Parameters(np.zeros((1, self.num_features, 1, 1)))
        self.gamma.name = "gamma"
        self.beta.name = "beta"

        # Running statistics
        self.running_mean: np.ndarray = np.zeros((1, self.num_features, 1, 1))
        self.running_var: np.ndarray = np.ones((1, self.num_features, 1, 1))

        # Cache for backward pass
        self.x: Optional[np.ndarray] = None
        self.x_hat: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self.x_mu: Optional[np.ndarray] = None
        self.m: Optional[int] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for 2D batch normalization.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Normalized and scaled output tensor.
        """
        self.x = x.astype(np.float32, copy=False)
        N, C, H, W = x.shape
        self.m = N * H * W  # Effective batch size

        if self._training:
            # Training mode: compute batch statistics
            mu = np.mean(x, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
            var_biased = np.var(x, axis=(0, 2, 3), keepdims=True)
            var_unbiased = (
                var_biased * (self.m / (self.m - 1)) if self.m > 1 else var_biased
            )

            x_mu = x - mu
            x_hat = x_mu / np.sqrt(var_unbiased + self.eps)

            # Cache for backward
            self.mu = mu
            self.var = var_unbiased
            self.x_mu = x_mu
            self.x_hat = x_hat

            out = self.gamma.data * x_hat + self.beta.data

            # Update running statistics
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mu
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var_unbiased

            return out
        else:
            # Evaluation mode: use running statistics
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma.data * x_hat + self.beta.data
            return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for 2D batch normalization.

        Args:
            grad: Gradient of loss with respect to output.

        Returns:
            Gradient of loss with respect to input.

        Raises:
            ValueError: If cached values from forward pass are missing.
        """
        if any(v is None for v in [self.x_hat, self.var, self.x_mu, self.m]):
            raise ValueError("Backward called before forward pass or cache cleared")

        # Local aliases for clarity
        m = self.m
        x_hat = self.x_hat
        var = self.var
        x_mu = self.x_mu
        eps = self.eps

        # Parameter gradients (reduce over batch and spatial dimensions)
        self.gamma.grad = np.sum(grad * x_hat, axis=(0, 2, 3), keepdims=True)
        self.beta.grad = np.sum(grad, axis=(0, 2, 3), keepdims=True)

        # Gradient through normalization
        dx_hat = grad * self.gamma.data

        inv_std = (var + eps) ** (-0.5)
        inv_std3 = (var + eps) ** (-1.5)

        # Gradient through variance
        dvar = np.sum(dx_hat * x_mu * (-0.5) * inv_std3, axis=(0, 2, 3), keepdims=True)
        # Gradient through mean
        dmu = (
            np.sum(dx_hat * (-inv_std), axis=(0, 2, 3), keepdims=True)
            + dvar * np.sum(-2.0 * x_mu, axis=(0, 2, 3), keepdims=True) / m
        )
        # Gradient through input
        dx = dx_hat * inv_std + dvar * (2.0 * x_mu) / m + dmu / m

        return dx

    def parameters(self) -> ListOfParameters:
        """Return learnable parameters.

        Returns:
            List containing gamma and beta Parameters objects.
        """
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"BatchNorm2d(num_features={self.num_features}, momentum={self.momentum}, eps={self.eps})"
