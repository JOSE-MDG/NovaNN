import numpy as np

from src.module.module import Parameters
from typing import Iterable, List

"""
Stochastic Gradient Descent optimizer with optional momentum and weight decay.

Improvements made (non-breaking):
- Materialize `parameters` iterable to a list
- Skip parameters with `grad is None`.
- Skip normalization params named 'gamma'/'beta'.
- Add type hints and concise Google-style docstrings.
"""


class SGD:
    """SGD optimizer.

    Args:
        parameters: Iterable of Parameters objects.
        learning_rate: Step size.
        momentum: Momentum coefficient (default 0 = no momentum).
        weight_decay: L2/L1 weight decay coefficient.
        lambda_l1: If True use L1 weight decay, otherwise L2.
    """

    def __init__(
        self,
        parameters: Iterable[Parameters],
        learning_rate: float,
        momentum: float = 0,
        weight_decay: float = 0,
        lambda_l1: bool = False,
    ) -> None:
        self.params: List[Parameters] = list(parameters)
        self.lr: float = float(learning_rate)
        self.beta: float = float(momentum)
        self.wd: float = float(weight_decay)
        self.l1: bool = bool(lambda_l1)
        # Initialize velocities matching parameter shapes
        self.velocities: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Perform a single optimization step."""
        for i, p in enumerate(self.params):
            # Skip batch-norm / normalization params or params without gradient
            if getattr(p, "name", None) in ("gamma", "beta"):
                continue
            if p.grad is None:
                continue

            # Apply weight decay (L1 or L2) to the gradient if configured
            if self.wd > 0:
                if self.l1:
                    p.grad = p.grad + self.wd * np.sign(p.data)
                else:
                    p.grad = p.grad + self.wd * p.data

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
