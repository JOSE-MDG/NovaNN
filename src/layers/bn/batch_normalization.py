import numpy as np

from src.layers.activations.activations import Activation
from src.module.module import Parameters


class BatchNormalization(Activation):
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.gamma = Parameters(np.ones((num_features, 1)))
        self.beta = Parameters(np.zeros((num_features, 1)))
        self.beta.name = "beta"
        self.gamma.name = "gamma"

        self.running_mean = np.zeros((num_features, 1))
        self.running_var = np.ones((num_features, 1))

        self.x = None
        self.x_hat = None
        self.mu = None
        self.var = None
        self.x_mu = None
        self.m = None

    def eval(self):
        self._training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.m = x.shape[1]
        if self._training:
            mu = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True)

            x_mu = x - mu
            x_hat = x_mu / np.sqrt(var + self.eps)

            self.mu = mu
            self.var = var
            self.x_mu = x_mu
            self.x_hat = x_hat

            out = self.gamma.data * x_hat + self.beta.data

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )

            return out
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma.data * x_hat + self.beta.data
            return out

    def backward(self, grad: np.ndarray) -> np.ndarray:

        m = self.m
        x_hat = self.x_hat
        var = self.var
        x_mu = self.x_mu
        eps = self.eps

        self.gamma.grad = np.sum(grad * x_hat, axis=1, keepdims=True)
        self.beta.grad = np.sum(grad, axis=1, keepdims=True)

        dx_hat = grad * self.gamma.data

        inv_std = (var + eps) ** (-0.5)  # 1 / sqrt(var + eps)
        inv_std3 = (var + eps) ** (-1.5)  # 1 / (var + eps)^(3/2)

        dvar = np.sum(dx_hat * x_mu * (-0.5) * inv_std3, axis=1, keepdims=True)
        dmu = np.sum(dx_hat * (-inv_std), axis=1, keepdims=True)

        dx = dx_hat * inv_std + dvar * (2 * x_mu) / m + dmu / m

        return dx

    def parameters(self):
        return [self.gamma, self.beta]
