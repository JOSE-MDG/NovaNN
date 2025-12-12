import numpy as np
import novann as nn
import novann.functional as F

from typing import Optional, Tuple


class CrossEntropyLoss:
    """Categorical cross-entropy loss for integer class labels.

    This implementation expects `logits` of shape (N, C) and `targets` as
    integer class indices (shape (N,)). The loss uses a numerically stable
    softmax followed by the negative log-likelihood with a small epsilon
    to avoid log(0).

    Attributes:
        N: Batch size observed in the last forward pass.
        y_hat: Cached softmax probabilities from the forward pass.
        y_one_hot: Cached one-hot encoded targets.
        eps: Small epsilon used in log to avoid numerical issues.
    """

    def __init__(self) -> None:
        self.N: int = 0
        self.y_hat: Optional[np.ndarray] = None
        self.y_one_hot: Optional[np.ndarray] = None
        self.eps: float = 1e-12

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute loss and cache softmax / one-hot targets.

        Args:
            logits: Raw model outputs, shape (N, C).
            targets: Integer class labels, shape (N,).

        Returns:
            Scalar loss (float).
        """
        self.N = logits.shape[0]
        C = logits.shape[1]

        # support targets already being one-hot or class indices
        if targets.ndim == 1 or targets.shape[1:] == ():
            y = np.eye(C, dtype=np.int32)[targets]
        else:
            y = targets.astype(np.int32)

        self.y_one_hot = y

        # numerically stable softmax
        self.y_hat = F.softmax(logits, dim=1)

        # Cross entropy formula
        loss = F.cross_entropy(self.y_hat, self.y_one_hot)
        return np.float32(loss)

    def backward(self) -> np.ndarray:
        """Return gradient of loss w.r.t. logits.

        Returns:
            Gradient array of shape (N, C).
        """
        if self.y_hat is None or self.y_one_hot is None:
            raise RuntimeError("forward must be called before backward")

        grad = (self.y_hat - self.y_one_hot) / self.N
        return grad.astype(np.float32)

    def __call__(
        self, logits: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Convenience: compute loss and gradient in one call."""
        cost = self.forward(logits=logits, targets=targets)
        grad = self.backward()
        return cost, grad


class MSE:
    """Mean squared error loss.

    Expects logits and targets with identical shapes. Returns averaged MSE.
    """

    def __init__(self) -> None:
        self.logits: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> np.float32:
        self.logits = logits
        self.targets = targets
        self.N: int = logits.shape[0]
        return F.mse_loss(logits, targets)

    def backward(self) -> np.ndarray:
        if self.logits is None or self.targets is None:
            raise RuntimeError("forward must be called before backward")
        grad = (2.0 / self.N) * (self.logits - self.targets)
        return grad.astype(np.float32)

    def __call__(
        self, logits: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        loss = self.forward(logits=logits, targets=targets)
        grad = self.backward()
        return loss, grad


class MAE:
    """Mean absolute error loss."""

    def __init__(self) -> None:
        self.logits: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self.logits = logits
        self.targets = targets
        self.N: int = logits.shape[0]
        return F.l1_loss(logits, targets)

    def backward(self) -> np.ndarray:
        if self.logits is None or self.targets is None:
            raise RuntimeError("forward must be called before backward")
        grad = np.sign(self.logits - self.targets) / self.N
        return grad.astype(np.float32)

    def __call__(
        self, logits: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        loss = self.forward(logits=logits, targets=targets)
        grad = self.backward()
        return loss, grad


class BinaryCrossEntropy:
    """Binary cross-entropy loss for sigmoid outputs.

    Expects logits shaped (N, ...) and targets of the same shape containing
    0/1 labels. Uses Sigmoid internally and adds eps inside the log.
    """

    def __init__(self) -> None:
        self.y: Optional[np.ndarray] = None
        self.p: Optional[np.ndarray] = None
        self.N: int = 0
        self.eps: float = 1e-12

    def forward(self, probs: np.ndarray, targets: np.ndarray) -> float:
        self.N = int(probs.shape[0])
        self.y = targets
        self.p = probs

        # return loss
        return F.binary_cross_entropy(probs, targets)

    def backward(self) -> np.ndarray:
        if self.p is None or self.y is None:
            raise RuntimeError("forward must be called before backward")
        grad = (self.p - self.y) / self.N
        return grad.astype(np.float32)

    def __call__(
        self, probabilities: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        loss = self.forward(probs=probabilities, targets=targets)
        grad = self.backward()
        return loss, grad
