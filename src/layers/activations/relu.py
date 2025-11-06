import numpy as np

from src.module.layer import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class LeakyReLU(Layer):
    def __init__(self, alpha: float):
        super().__init__()
        self.a = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, x, self.a * x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(grad > 0, 1, self.a)
