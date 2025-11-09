import numpy as np

from src.layers.activations.activations import Activation


class ReLU(Activation):
    def __init__(self):
        super().__init__()
        self._mask = None
        self.affect_init = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class LeakyReLU(Activation):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.a = negative_slope
        self.affect_init = True
        self.activation_param = negative_slope

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache_intput = x
        return np.where(x >= 0, x, self.a * x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self._cache_intput
        return grad * np.where(x >= 0, 1, self.a)
