import numpy as np
from typing import Iterable, List
from novann._typing import ListOfParameters
from novann.module import Parameters


class RMSprop:
    """RMSProp optimizer.

    Args:
        parameters: Iterable of Parameters objects.
        lr: Step size.
        beta: Decay factor for moving average of squared gradients (default: 0.9)
        weight_decay: L2 decoupled weight decay coefficient.
        epsilon: arbitrarily small positive number
    """

    def __init__(
        self,
        parameters: Iterable[Parameters],
        lr: float,
        beta: float = 0.9,
        weight_decay: float = 0,
        epsilon: float = 1e-9,
    ) -> None:

        # Materialize parameters to a list
        self.params: ListOfParameters = list(parameters)
        self.lr: float = float(lr)
        self.beta: float = float(beta)
        self.wd: float = float(weight_decay)
        self.is_bn_param: bool = False

        # Initialize moment buffers to match each parameter's shape.
        self.moments: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self.eps: float = float(epsilon)

    def step(self) -> None:
        """Perform a single optimization step."""
        for i, p in enumerate(self.params):
            # Skip params without gradient or BN params (gamma/beta)
            if p.grad is None:
                continue

            # Update running average of squared gradients
            self.moments[i] = self.beta * self.moments[i] + (1.0 - self.beta) * (
                p.grad**2
            )

            # Parameter update
            p.data -= self.lr * (p.grad / (np.sqrt(self.moments[i]) + self.eps))

            # Verify that they are not batch normalization layer parameters
            self.is_bn_param = getattr(p, "name", None) in ("gamma", "beta")

            # apply decoupled weight decay (L2)
            if self.wd > 0 and not self.is_bn_param:
                p.data -= self.lr * self.wd * p.data  # L2

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero or clear gradients for all managed parameters."""
        for p in self.params:
            p.zero_grad(set_to_none=set_to_none)
