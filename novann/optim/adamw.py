import numpy as np
from typing import List
from novann._typing import BetaCoefficients, IterableParameters, ListOfParameters


class AdamW:
    """AdamW optimizer with decoupled weight decay.

    Args:
        parameters: Iterable of Parameters to optimize.
        lr: Learning rate.
        betas: Coefficients for momentum (beta1) and squared gradient (beta2).
        weight_decay: Decoupled L2 regularization coefficient.
        epsilon: Numerical stability term.
    """

    def __init__(
        self,
        parameters: IterableParameters,
        lr: float,
        betas: BetaCoefficients = (0.9, 0.999),
        weight_decay: float = 0,
        epsilon: float = 1e-8,
    ):
        # Materialize parameters to a list
        self.params: ListOfParameters = list(parameters)
        self.lr: float = float(lr)
        self.wd: float = float(weight_decay)

        # Beta coefficients
        self.b1: float = float(betas[0])
        self.b2: float = float(betas[1])
        self.eps: float = float(epsilon)

        # First and second moment buffers
        self.m_t: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self.v_t: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

        self.t: int = 0  # Time step counter

    def step(self):
        """Performs a single optimization step."""
        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Skip BatchNorm parameters for weight decay
            is_bn_param = getattr(p, "name", None) in ("gamma", "beta")

            # Update moment estimates
            self.m_t[i] = self.b1 * self.m_t[i] + (1 - self.b1) * p.grad
            self.v_t[i] = self.b2 * self.v_t[i] + (1 - self.b2) * (p.grad**2)

            # Bias correction
            m_hat = self.m_t[i] / (1 - self.b1**self.t)
            v_hat = self.v_t[i] / (1 - self.b2**self.t)

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Apply decoupled weight decay (skip BatchNorm parameters)
            if self.wd > 0 and not is_bn_param:
                p.data -= self.lr * self.wd * p.data

    def zero_grad(self, set_to_none: bool = False):
        """Clears gradients of all optimized parameters."""
        for p in self.params:
            p.zero_grad(set_to_none=set_to_none)
