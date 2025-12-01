import numpy as np
from typing import Optional, Tuple
from novann._typing import ListOfParameters

from novann.module import Layer, Parameters


class BatchNorm1d(Layer):
    """Batch Normalization layer for 1D/2D inputs (fully connected or 1D convolutional).

    Normalizes activations per feature across the batch dimension. Uses running
    estimates of mean and variance during evaluation and batch statistics during
    training. Includes learnable scale (gamma) and shift (beta) parameters.

    Args:
        num_features: Number of features/channels.
        momentum: Momentum for running statistics update (default: 0.1).
        eps: Epsilon for numerical stability (default: 1e-5).

    Attributes:
        gamma: Scale parameter (shape adapted to input dimensions).
        beta: Shift parameter (shape adapted to input dimensions).
        running_mean: Running mean estimate.
        running_var: Running variance estimate.
        x: Cached input from forward pass.
        x_hat: Normalized input.
        mu: Batch mean.
        var: Batch variance.
        x_mu: Input minus mean.
        m: Effective batch size for statistics.
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        """Initialize parameters, running statistics, and cache."""
        super().__init__()
        self.num_features: int = int(num_features)
        self.momentum: float = float(momentum)
        self.eps: float = float(eps)

        # Initialize parameters with proper shapes
        self.gamma: Parameters = Parameters(np.ones((1, self.num_features)))
        self.beta: Parameters = Parameters(np.zeros((1, self.num_features)))
        self.gamma.name = "gamma"
        self.beta.name = "beta"

        # Running statistics
        self.running_mean: np.ndarray = np.zeros((1, self.num_features))
        self.running_var: np.ndarray = np.ones((1, self.num_features))

        # Cache for backward pass
        self.x: Optional[np.ndarray] = None
        self.x_hat: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self.x_mu: Optional[np.ndarray] = None
        self.m: Optional[int] = None

    def _verify_dims(self, x: np.ndarray) -> Tuple[Tuple[int, ...], int]:
        """Validate input dimensions and compute statistics axes.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (axes for statistics computation, number of dimensions).

        Raises:
            ValueError: If input has invalid dimensions.
        """
        x = x.astype(np.float32, copy=False)
        self.x = x
        dims = x.ndim

        if dims == 2:  # (batch_size, features)
            axes = (0,)  # Average over batch dimension
            self.m = x.shape[0]  # Batch size
            return axes, dims
        elif dims == 3:  # (batch_size, channels, sequence_length)
            axes = (0, 2)  # Average over batch and sequence dimensions
            self.m = x.shape[0] * x.shape[2]  # Batch size * sequence length

            # Reshape parameters for 3D inputs
            self.gamma.data = self.gamma.data.reshape(1, self.num_features, 1)
            self.beta.data = self.beta.data.reshape(1, self.num_features, 1)
            return axes, dims
        else:
            raise ValueError(f"BatchNorm1d expects 2D or 3D input, got {dims}D")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for batch normalization.

        Args:
            x: Input tensor of shape (batch_size, features) or
               (batch_size, channels, sequence_length).

        Returns:
            Normalized and scaled output tensor.

        Raises:
            ValueError: If input dimensions are invalid.
        """
        axes, dims = self._verify_dims(x)

        if self._training:
            # Training mode: use batch statistics
            mu = np.mean(x, axis=axes, keepdims=True)
            var_biased = np.var(x, axis=axes, keepdims=True)
            # Bessel's correction for unbiased variance estimate
            var_unbiased = (
                var_biased * (self.m / (self.m - 1)) if self.m > 1 else var_biased
            )

            x_mu = x - mu
            x_hat = x_mu / np.sqrt(var_unbiased + self.eps)

            # Cache values for backward pass
            self.mu = mu
            self.var = var_unbiased
            self.x_mu = x_mu
            self.x_hat = x_hat

            # Scale and shift
            out = self.gamma.data * x_hat + self.beta.data

            # Update running statistics
            mu_flat = mu.reshape(1, self.num_features)
            var_flat = var_unbiased.reshape(1, self.num_features)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mu_flat
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var_flat

            return out

        else:
            # Evaluation mode: use running statistics
            if dims == 3:
                rm = self.running_mean.reshape(1, self.num_features, 1)
                rv = self.running_var.reshape(1, self.num_features, 1)
            else:
                rm = self.running_mean
                rv = self.running_var

            x_hat = (x - rm) / np.sqrt(rv + self.eps)
            out = self.gamma.data * x_hat + self.beta.data
            return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients for parameters and inputs.

        Computes gradients for gamma and beta parameters and returns
        gradient with respect to the input.

        Args:
            grad: Gradient of loss with respect to output, same shape as input.

        Returns:
            Gradient of loss with respect to input, same shape as grad.

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

        # Determine axes for reduction based on input dimensions
        dims = self.x.ndim
        if dims == 2:
            axes = (0,)
        elif dims == 3:
            axes = (0, 2)
        else:
            raise ValueError(f"Unexpected input dimensionality: {dims}")

        # Compute parameter gradients
        self.gamma.grad = np.sum(grad * x_hat, axis=axes, keepdims=True)
        self.beta.grad = np.sum(grad, axis=axes, keepdims=True)

        # Gradient through normalization
        dx_hat = grad * self.gamma.data

        inv_std = (var + eps) ** (-0.5)
        inv_std3 = (var + eps) ** (-1.5)

        # Gradient through variance
        dvar = np.sum(dx_hat * x_mu * (-0.5) * inv_std3, axis=axes, keepdims=True)
        # Gradient through mean
        dmu = (
            np.sum(dx_hat * (-inv_std), axis=axes, keepdims=True)
            + dvar * np.sum(-2.0 * x_mu, axis=axes, keepdims=True) / m
        )
        # Gradient through input
        dx = dx_hat * inv_std + dvar * (2.0 * x_mu) / m + dmu / m

        return dx

    def parameters(self) -> ListOfParameters:
        """Return learnable parameters (gamma and beta).

        Returns:
            List containing gamma and beta Parameters objects.
        """
        return [self.gamma, self.beta]


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
