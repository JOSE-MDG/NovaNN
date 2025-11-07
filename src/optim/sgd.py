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

    def step(self):
        for p in self.params:
            grad = p.grad.copy()
            if self.wd > 0:
                pass
