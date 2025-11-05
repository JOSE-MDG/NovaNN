import numpy as np
from typing import Iterable


class Parameters:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        self.grad.fill(0.0)


class Module:
    def __init__(self):
        self._training = True

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def parameters(self) -> Iterable[Parameters]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
