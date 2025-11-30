import numpy as np
from src.module import Layer


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self._origin_shape = None

    def forward(self, x) -> np.ndarray:
        self._origin_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(*self._origin_shape)
