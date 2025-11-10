import numpy as np
from typing import Iterable, Tuple
from src.module.module import Parameters


class Adam:
    def __init__(
        self,
        parameters: Iterable[Parameters],
        learning_rate: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0,
        lambda_l1: bool = False,
        epsilon: float = 1e-9,
    ):
        self.params = parameters
        self.lr = learning_rate
        self.wd = weight_decay
        self.l1 = lambda_l1
        self.moments = [np.zeros_like(p.data) for p in parameters]
        self.velocities = [np.zeros_like(p.data) for p in parameters]
        self.b1 = betas[0]
        self.b2 = betas[1]
        self.eps = epsilon
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if getattr(p, "name", None) is None and self.wd > 0:
                p.grad += self.wd * np.sign(p.data) if self.l1 else self.wd * p.data

            self.velocities[i] = self.b1 * self.velocities[i] + (1 - self.b1) * p.grad
            self.moments[i] = self.b2 * self.moments[i] + (1 - self.b2) * (p.grad**2)

            m_hat = self.velocities[i] / (1 - self.b1**self.t)
            v_hat = self.moments[i] / (1 - self.b2**self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, set_to_none: bool = False):
        for p in self.params:
            p.zero_grad(set_to_none=set_to_none)
