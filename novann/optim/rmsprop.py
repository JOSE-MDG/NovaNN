import numpy as np
from typing import List
from novann._typing import ListOfParameters, IterableParameters, BetaCoefficients


class RMSprop:
    """RMSprop optimizer with decoupled weight decay.

    Args:
        parameters: Iterable of Parameters to optimize.
        lr: Learning rate.
        beta: Decay factor for moving average of squared gradients.
        weight_decay: Decoupled L2 regularization coefficient.
        epsilon: Numerical stability term.
    """

    def __init__(
        self,
        parameters: IterableParameters,
        lr: float,
        beta: BetaCoefficients = 0.9,  # Corregí el tipo: BetaCoefficients -> float
        weight_decay: float = 0,
        epsilon: float = 1e-9,
    ) -> None:
        # Materialize parameters to a list
        self.params: ListOfParameters = list(parameters)
        self.lr: float = float(lr)
        self.beta: BetaCoefficients = float(beta)
        self.wd: float = float(weight_decay)
        self.eps: float = float(epsilon)

        # Initialize squared gradient buffers
        self.moments: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Performs a single optimization step."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Update running average of squared gradients
            self.moments[i] = self.beta * self.moments[i] + (1.0 - self.beta) * (
                p.grad**2
            )

            # RMSprop update
            p.data -= self.lr * (p.grad / (np.sqrt(self.moments[i]) + self.eps))

            # Apply decoupled weight decay (skip BatchNorm parameters)
            is_bn_param = getattr(p, "name", None) in ("gamma", "beta")
            if self.wd > 0 and not is_bn_param:
                p.data -= self.lr * self.wd * p.data  # Decoupled L2

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clears gradients of all optimized parameters."""
        for p in self.params:
            p.zero_grad(set_to_none=set_to_none)
