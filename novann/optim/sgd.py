import numpy as np

from novann.module import Parameters
from typing import Iterable, List, Optional
from novann._typing import ListOfParameters


class SGD:
    """SGD optimizer.

    Args:
        parameters: Iterable of Parameters objects.
        lr: Step size.
        momentum: Momentum coefficient (default 0 = no momentum).
        weight_decay: L2 weight decay coefficient.
        max_grad_norm: If set to a positive value, clip each parameter's
            gradient to have at most this L2 norm. Useful to avoid exploding
            gradients / numerical overflows.
    """

    def __init__(
        self,
        parameters: Iterable[Parameters],
        lr: float,
        momentum: float = 0,
        weight_decay: float = 0,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.params: ListOfParameters = list(parameters)
        self.lr: float = float(lr)
        self.beta: float = float(momentum)
        self.wd: float = float(weight_decay)
        self.is_bn_param: bool = False
        self.max_grad_norm: Optional[float] = (
            float(max_grad_norm)
            if max_grad_norm is not None and max_grad_norm > 0
            else None
        )

        # Initialize velocities matching parameter shapes
        self.velocities: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def _global_clipping(self) -> float:
        """Global clipping

        Global Clipping is a safety mechanism that ensures the model never takes
        an optimization step longer than allowed, no matter how much the gradient
        has "screamed".

        Args:
            parameter (Parameters): Parameter object

        Returns:
            float: A global scale factor
        """
        if self.max_grad_norm is None:
            return 1.0

        total_norm_sq = 0.0
        # Calculate the overall norm of ALL gradients first
        for p in self.params:
            if p.grad is None:
                continue
            # We add the L2 norm to the square of each parameter
            total_norm_sq += np.sum(p.grad**2)

        total_norm = np.sqrt(total_norm_sq)

        # We calculate the global scale factor
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)

        # If the coefficient is < 1, it means it exploded and needs to be reduced.
        # If it's > 1, we do nothing (clamp max=1.0) to avoid enlarging small gradients.
        clip_coef = min(clip_coef, 1.0)

        return clip_coef

    def step(self) -> None:
        """Perform a single optimization step."""
        # Calculate the scale coefficient
        clip_coef = self._global_clipping()
        for i, p in enumerate(self.params):
            # Skip params without gradient or BN params (gamma/beta)
            if p.grad is None:
                continue

            # Apply Global Clipping immediately
            p.grad *= clip_coef

            # Verify that they are not batch normalization layer parameters
            self.is_bn_param = getattr(p, "name", None) in ("gamma", "beta")

            # Apply coupled weight decay to de gradient
            if self.wd > 0 and not self.is_bn_param:
                p.grad += self.wd * p.data  # L2

            # Momentum update
            if self.beta > 0:
                self.velocities[i] = self.beta * self.velocities[i] - self.lr * p.grad
                p.data += self.velocities[i]
            else:
                p.data -= self.lr * p.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero or clear gradients for all managed parameters."""
        for p in self.params:
            p.zero_grad(set_to_none=set_to_none)
