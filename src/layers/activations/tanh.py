import numpy as np
from src.layers.activations.activations import Activation


class Tanh(Activation):
    def __init__(self):
        super().__init__()
        self.affect_init = True
        self.out = None

    def forward(self, x: np.ndarray):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1 - self.out**2)
