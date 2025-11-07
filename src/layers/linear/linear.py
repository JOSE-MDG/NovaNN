import numpy as np

from src.module.module import Parameters
from src.module.layer import Layer
from typing import Optional, Callable


class Linear(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: Optional[Callable] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = None
        self.b = bias
        self.init_fn = init
        self._cache_input = None
        self.reset_parameters()

    def reset_parameters(self, initializer: Optional[Callable] = None): ...

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache_input = x
        out = x @ self.weight.data.T
        if self.bias is not None:
            out += self.bias.data
        return out

    def backward(self, grad) -> np.ndarray:
        x = self._cache_input
        self.weight.grad = grad.T @ x
        if self.bias is not None:
            self.bias.grad = np.sum(grad, axis=1, keepdims=True)
        grad_input = grad @ self.weight.data
        return grad_input

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
