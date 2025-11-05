import numpy as np
from src.module.layer import Layer
from src.module.module import Parameters
from typing import List, Iterable


class Sequential:
    def __init__(self, modules: List[Layer]):
        super().__init__()
        self.layers = modules
        self._modules = modules

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()

    def forward(self, x: np.ndarray) -> np.ndarray:
        input = x
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad_input = grad
        for layer in reversed(self.layers):
            grad_input = layer.backward(grad_input)
        return grad_input

    def parameters(self) -> Iterable[Parameters]:
        parameters = []
        for layer in self.layers:
            parameters.append(list(layer.parameters()))
        return parameters

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
