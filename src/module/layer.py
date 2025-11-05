import numpy as np
from src.module.module import Module


class Layer(Module):
    def __init__(self):
        super().__init__()
        self._have_params = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
