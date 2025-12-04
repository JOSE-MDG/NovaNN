import numpy as np

from novann.module.module import Parameters
from typing import Iterable, List, Optional


class SGD:
    """SGD optimizer.

    Args:
        parameters: Iterable of Parameters objects.
        lr: Step size.
        momentum: Momentum coefficient (default 0 = no momentum).
        weight_decay: L2/L1 weight decay coefficient.
        lambda_l1: If True use L1 weight decay, otherwise L2.
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
        lambda_l1: bool = False,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.params: List[Parameters] = list(parameters)
        self.lr: float = float(lr)
        self.beta: float = float(momentum)
        self.wd: float = float(weight_decay)
        self.l1: bool = lambda_l1
        self.is_bn_param: bool = False
        self.max_grad_norm: Optional[float] = (
            float(max_grad_norm)
            if max_grad_norm is not None and max_grad_norm > 0
            else None
        )

        # Initialize velocities matching parameter shapes
        self.velocities: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def _clip_grad_inplace(self, grad: np.ndarray) -> None:
        """Clip gradient in-place to respect `max_grad_norm`, if enabled."""
        if self.max_grad_norm is None:
            return

        # Flatten to compute L2 norm, then rescale if needed
        norm = float(np.linalg.norm(grad))
        if norm == 0.0 or norm <= self.max_grad_norm:
            return

        scale = self.max_grad_norm / (norm + 1e-12)
        grad *= scale

    def step(self) -> None:
        """Perform a single optimization step."""
        for i, p in enumerate(self.params):
            # Skip params without gradient or BN params (gamma/beta)
            if p.grad is None:
                continue

            self.is_bn_param = getattr(p, "name", None) in ("gamma", "beta")

            # Apply weight decay (L1 or L2) to the gradient
            if self.wd > 0 and not self.is_bn_param:
                if self.l1:
                    p.grad += self.wd * np.sign(p.data)
                else:
                    p.grad += self.wd * p.data

            # Clip gradients if requested to avoid exploding updates
            self._clip_grad_inplace(p.grad)

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
