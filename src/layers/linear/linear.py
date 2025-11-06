import numpy as np

from src.module.module import Parameters
from src.module.layer import Layer


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = Parameters(None)
        self.bias = Parameters(np.zeros((1, out_features))) if bias else None
        self._cache_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache_input = x
        out = x @ self.weight.data.T
        if self.bias is not None:
            out += self.bias
        return out

    def backward(self, grad) -> np.ndarray:
        x = self._cache_input
        self.weight.grad = grad.T @ x
        if self.bias is not None:
            self.bias.grad = np.sum(grad, axis=0, keepdims=True)
        grad_input = grad @ self.weight.data
        return grad_input

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
