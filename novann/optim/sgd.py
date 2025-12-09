import numpy as np
from typing import List, Optional
from novann._typing import ListOfParameters, IterableParameters, BetaCoefficients


class SGD:
    """SGD optimizer with momentum and gradient clipping.

    Args:
        parameters: Iterable of Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum coefficient (0 for vanilla SGD).
        weight_decay: L2 regularization coefficient.
        max_grad_norm: Maximum L2 norm for gradient clipping.
    """

    def __init__(
        self,
        parameters: IterableParameters,
        lr: float,
        momentum: float = 0,
        weight_decay: float = 0,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        # Materialize parameters to a list
        self.params: ListOfParameters = list(parameters)
        self.lr: float = float(lr)
        self.wd: float = float(weight_decay)
        self.beta: BetaCoefficients = float(momentum)
        self.max_grad_norm: Optional[float] = (
            float(max_grad_norm) if max_grad_norm else None
        )

        # Velocity buffers for momentum
        self.velocities: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def _global_clipping(self) -> float:
        """Computes global gradient clipping coefficient.

        Returns:
            Clipping coefficient (1.0 if no clipping needed).
        """
        if self.max_grad_norm is None:
            return 1.0

        # Calculate total gradient norm
        total_norm_sq = 0.0
        for p in self.params:
            if p.grad is not None:
                total_norm_sq += np.sum(p.grad**2)

        total_norm = np.sqrt(total_norm_sq)

        # Compute clipping coefficient
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        return min(clip_coef, 1.0)

    def step(self) -> None:
        """Performs a single optimization step."""
        clip_coef = self._global_clipping()

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Apply gradient clipping
            p.grad *= clip_coef

            # Skip BatchNorm parameters for weight decay
            is_bn_param = getattr(p, "name", None) in ("gamma", "beta")

            # Apply weight decay to gradient
            if self.wd > 0 and not is_bn_param:
                p.grad += self.wd * p.data  # L2 coupled

            # Momentum update
            if self.beta > 0:
                self.velocities[i] = self.beta * self.velocities[i] - self.lr * p.grad
                p.data += self.velocities[i]
            else:
                # Vanilla SGD update
                p.data -= self.lr * p.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clears gradients of all optimized parameters."""
        for p in self.params:
            p.zero_grad(set_to_none=set_to_none)
