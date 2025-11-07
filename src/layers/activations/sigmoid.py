import numpy as np
from src.layers.activations.activations import Activation


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.affect_init = True
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = grad * (self.out * (1 - self.out))
