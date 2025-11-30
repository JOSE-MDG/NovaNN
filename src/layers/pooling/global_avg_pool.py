import numpy as np
from src.module import Layer


class GlobalAvgPool1d(Layer):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)  # (N,C,L)

        output = x.mean(axis=2, keepdims=True)
        self._cache["x_shape"] = x.shape
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:

        grad_output = grad_output.astype(np.float32, copy=False)
        N, C, L = self._cache["x_shape"]
        factor = 1.0 / L
        grad_input = np.ones((N, C, L), dtype=np.float32) * (grad_output * factor)
        return grad_input


class GlobalAvgPool2d(Layer):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)

        output = x.mean(axis=(2, 3), keepdims=True)
        self._cache["x_shape"] = x.shape
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_output = grad_output.astype(np.float32, copy=False)
        N, C, H, W = self._cache["x_shape"]
        factor = 1.0 / (H * W)
        grad_input = np.ones((N, C, H, W), dtype=np.float32) * (grad_output * factor)
        return grad_input
