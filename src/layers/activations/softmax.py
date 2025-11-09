import numpy as np

from src.layers.activations.activations import Activation


class SoftMax(Activation):
    def __init__(self, axis: int = 0):
        super().__init__()
        self.affect_init = False
        self.out = None
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits_max = np.max(x, axis=self.axis, keepdims=True)
        stable_logits = x - logits_max
        logits_exp = np.exp(stable_logits)
        logits_sum = np.sum(logits_exp, axis=self.axis, keepdims=True)
        self.out = logits_exp / logits_sum
        return self.out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        s = np.sum(self.out * grad, axis=self.axis, keepdims=True)
        grad = self.out * (grad - s)
        return grad
