import numpy as np

from src.module.module import Parameters
from typing import Iterable


class SGD:
    def __init__(
        self,
        parameters: Iterable[Parameters],
        learning_rate: float,
        momentum: float = 0,
        weight_decay: float = 0,
        lambda_l1: bool = False,
    ):
        self.params = parameters
        self.lr = learning_rate
        self.beta = momentum
        self.wd = weight_decay
        self.l1 = lambda_l1
        self.velocities = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for i, p in enumerate(self.params):
            if getattr(p, "name", None) is None and self.wd > 0:
                p.grad += self.wd * np.sign(p.data) if self.l1 else self.wd * p.data
            if self.beta > 0:
                self.velocities[i] = self.beta * self.velocities[i] - self.lr * p.grad
                p.data += self.velocities[i]
            else:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
