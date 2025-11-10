import numpy as np
from src.layers.activations.softmax import SoftMax
from src.layers.activations.sigmoid import Sigmoid


class CrossEntropyLoss:
    def __init__(self):
        self.N = 0
        self.y_hat = None
        self.y_one_hot = None
        self.eps = 1e-12

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self.N = logits.shape[0]

        C = logits.shape[1]
        y = np.eye(C)[targets]
        self.y_one_hot = y

        y_hat = SoftMax().forward(logits)
        self.y_hat = y_hat

        loss = -np.sum(self.y_one_hot * np.log(self.y_hat + self.eps)) / self.N

        return loss

    def backward(self) -> np.ndarray:
        grad = self.y_hat - self.y_one_hot
        grad /= self.N
        return grad

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        self.cost = self.forward(logits=logits, targets=targets)
        grad = self.backward()

        return self.cost, grad


class MSE:
    def __init__(self):
        self.logits = None
        self.targets = None
        self.N = 0

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self.N = logits.shape[0]
        self.logits = logits
        self.targets = targets
        return np.sum((logits - targets) ** 2) / self.N

    def backward(self) -> np.ndarray:
        grad = (2 / self.N) * (self.logits - self.targets)
        return grad

    def __call__(self, logits, targets) -> np.ndarray:
        loss = self.forward(logits=logits, targets=targets)
        grad = self.backward()
        return loss, grad


class MAE:
    def __init__(self):
        self.logits = None
        self.targets = None
        self.N = 0

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self.N = logits.shape[0]
        self.logits = logits
        self.targets = targets
        return np.mean(np.abs(logits - targets))

    def backward(self) -> np.ndarray:
        grad = np.sign(self.logits - self.targets)
        grad /= self.N
        return grad

    def __call__(self, logits, targets) -> np.ndarray:
        loss = self.forward(logits=logits, targets=targets)
        grad = self.backward()
        return loss, grad


class BinaryCrossEntropy:
    def __init__(self):
        self.y = None
        self.y_hat = None
        self.N = 0
        self.eps = 1e-12

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self.N = logits.shape[0]
        self.y = targets
        self.y_hat = Sigmoid().forward(logits)

        loss = -np.mean(
            self.y * np.log(self.y_hat + self.eps)
            + (1 - self.y) * np.log(1 - self.y_hat + self.eps)
        )
        return loss

    def backward(self) -> np.ndarray:
        grad = self.y_hat - self.y
        grad /= self.N
        return grad

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        loss = self.forward(logits=logits, targets=targets)
        grad = self.backward()
        return loss, grad
