import numpy as np

from src.module.module import Parameters
from typing import Iterable


class RMSProp:
    def __init__(
        self,
        parameters: Iterable[Parameters],
        learning_rate: float,
        beta: float = 0.9,
        weight_decay: float = 0,
        lambda_l1: bool = False,
        epsilon: float = 1e-9,
    ):
        self.params = parameters
        self.lr = learning_rate
        self.beta = beta
        self.wd = weight_decay
        self.l1 = lambda_l1
        self.moments = [np.zeros_like(p.data) for p in parameters]
        self.eps = epsilon

    def step(self):
        for i, p in enumerate(self.params):
            if self.wd > 0:
                p.grad += self.wd * np.sign(p.data) if self.l1 else self.wd * p.data
            self.moments[i] = self.beta * self.moments[i] + (1 - self.beta) * (
                p.grad**2
            )
            p.data -= self.lr * (p.grad / np.sqrt(self.moments[i]) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
