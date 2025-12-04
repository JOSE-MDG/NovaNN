import numpy as np
from typing import Iterable, Tuple, List
from novann.module.module import Parameters


class Adam:
    """Adam optimizer.

    Args:
        parameters: Iterable of Parameters objects.
        lr: Step size.
        betas: Coefficients for computing running averages of gradient and its square.
               First value (beta1) is for the gradient, second (beta2) for squared gradient.
               Default: (0.9, 0.999).
        weight_decay: L2/L1 weight decay coefficient.
        lambda_l1: If True use L1 weight decay, otherwise L2.
        epsilon: Small constant for numerical stability.
    """

    def __init__(
        self,
        parameters: Iterable[Parameters],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0,
        lambda_l1: bool = False,
        epsilon: float = 1e-8,
    ) -> None:

        # Materialize parameters to a list
        self.params: List[Parameters] = list(parameters)
        self.lr: float = float(lr)
        self.wd: float = float(weight_decay)
        self.l1: bool = float(lambda_l1)

        # First and second moment buffers
        self.moments: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self.velocities: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self.b1: float = float(betas[0])
        self.b2: float = float(betas[1])
        self.eps: float = epsilon
        self.is_bn_param: bool = False
        self.t: int = 0

    def step(self) -> None:
        """Perform a single optimization step over the provided parameters."""
        self.t += 1
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

            # Update first and second moment estimates
            self.velocities[i] = self.b1 * self.velocities[i] + (1 - self.b1) * p.grad
            self.moments[i] = self.b2 * self.moments[i] + (1 - self.b2) * (p.grad**2)

            # Bias-corrected estimates
            m_hat = self.velocities[i] / (1 - self.b1**self.t)
            v_hat = self.moments[i] / (1 - self.b2**self.t)

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero or clear gradients for all parameters managed by this optimizer."""
        for p in self.params:
            p.zero_grad(set_to_none=set_to_none)
