import numpy as np

from src.layers.activations.activations import Activation


class BatchNormalization(Activation):
    def __init__(self, dim: int, momentum: float = 0.9, epsilon: float = 1e-12):
        super().__init__()
        self.momentum = momentum
        self.eps = epsilon
        self.affect_init = False

        self.gamma = np.zeros((1, dim))
        self.beta = None

        self.running_mean = None
        self.running_var = None

    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def backward(self, grad: np.ndarray) -> np.ndarray: ...
