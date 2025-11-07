import numpy as np

from src.layers.activations.activations import Activation


class SoftMax(Activation):
    def __init__(self):
        super().__init__()
        self.affect_init = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits_max = np.max(x, axis=1, keepdims=True)
        stable_logits = x - logits_max
        logits_exp = np.exp(stable_logits)
        logits_sum = np.sum(logits_exp, axis=1, keepdims=True)
        return logits_exp / logits_sum

    def backward(self, grad: np.ndarray) -> np.ndarray: ...
