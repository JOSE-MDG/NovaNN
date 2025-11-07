import numpy as np
from src.module.layer import Layer


class Dropout(Layer):
    def __init__(self, p: float):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError("Dropout probability must bebetween 0 and 1")
        self.p = p
        self._mask = None

    def eval(self):
        self._training = False
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._training:
            self._mask = (np.random.randn(*x.shape) > self.p).astype(np.float32)
            return (x * self._mask) / (1 - self.p)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._training and self._mask is not None:
            return (grad * self._mask) / (1 - self.p)
        return grad
