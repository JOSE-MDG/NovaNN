import numpy as np
from typing import Iterable, List
from src.module.module import Parameters


class RMSprop:
    """RMSProp optimizer (simple reference implementation).

    Notes:
        - Accepts an iterable of Parameters; the iterable is materialized to a
          list so multiple passes are safe.
        - Skips parameters with `grad is None` and parameters named 'gamma'/'beta'.
    """

    def __init__(
        self,
        parameters: Iterable[Parameters],
        learning_rate: float,
        beta: float = 0.9,
        weight_decay: float = 0,
        lambda_l1: bool = False,
        epsilon: float = 1e-9,
    ) -> None:

        # Materialize parameters to a list
        self.params: List[Parameters] = list(parameters)
        self.lr: float = float(learning_rate)
        self.beta: float = float(beta)
        self.wd: float = float(weight_decay)
        self.l1: bool = bool(lambda_l1)
        self.is_bn_param: bool = False

        # Initialize moment buffers to match each parameter's shape.
        self.moments: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self.eps: float = float(epsilon)

    def step(self) -> None:
        """Perform a single optimization step."""
        for i, p in enumerate(self.params):
            # Skip normalization params or parameters without gradient
            if getattr(p, "name", None) in ("gamma", "beta"):
                continue
            if p.grad is None:
                continue

            # Skip params without gradient or BN params (gamma/beta)
            self.is_bn_param = getattr(p, "name", None) in ("gamma", "beta")

            # Apply weight decay (L1 or L2) to the gradient
            if self.wd > 0 and not self.is_bn_param:
                if self.l1:
                    p.grad = p.grad + self.wd * np.sign(p.data)
                else:
                    p.grad = p.grad + self.wd * p.data

            # Update running average of squared gradients
            self.moments[i] = self.beta * self.moments[i] + (1.0 - self.beta) * (
                p.grad**2
            )

            # Parameter update
            p.data -= self.lr * (p.grad / (np.sqrt(self.moments[i]) + self.eps))

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero or clear gradients for all managed parameters."""
        for p in self.params:
            p.zero_grad(set_to_none=set_to_none)
