import numpy as np

from src.layers.activations.activations import Activation


class BatchNormalization(Activation):
    def __init__(self, dim: int, momentum: float = 0.9, epsilon: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = epsilon
        self.affect_init = False

        self.gamma = np.ones((dim, 1))
        self.beta = np.zeros((dim, 1))

        self.running_var = np.ones((dim, 1))
        self.running_mean = np.zeros((dim, 1))

        self._cache_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache_input = x
        if self._training:
            self.mu = np.mean(x, axis=1, keepdims=True)
            self.var = np.var(x, axis=1, keepdims=True)
            self.x_hat = (x - self.mu) / np.sqrt(self.var + self.eps)
            out = self.gamma * self.x_hat + self.beta

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            )
            self.runninf_var = (
                self.momentum * self.running_var + (1 - self.momentum) * self.var
            )
        else:
            x_hat = (x - self.mu) / np.sqrt(self.var + self.eps)
            out = self.gamma * x_hat + self.beta
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        B = grad.shape[1]  # (in,B)

        dbeta = np.sum(grad, axis=1, keepdims=True)
        dgamma = np.sum(grad * self.x_hat, axis=1, keepdims=True)

        dx_hat = grad * self.gamma

        dvar = np.sum(
            dx_hat * (self.x - self.mu) * -0.5 * (self.var + self.eps) ** (-1.5),
            axis=1,
            keepdims=True,
        )
        dmu = np.sum(
            dx_hat * -1 / np.sqrt(self.var + self.eps), axis=1, keepdims=True
        ) + dvar * np.mean(-2 * (self.x - self.mu), axis=1, keepdims=True)

        dx = (
            dx_hat / np.sqrt(self.var + self.eps)
            + dvar * 2 * (self.x - self.mu) / B
            + dmu / B
        )

        self.grad_gamma = dgamma
        self.grad_beta = dbeta
        return dx
