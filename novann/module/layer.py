import numpy as np
from abc import ABC, abstractmethod
from novann.module import Module


class Layer(Module, ABC):
    """Abstract base class for all neural network layers.

    Inherits from Module and defines the interface that all layers must
    implement. Subclasses must override `forward` and `backward` methods.

    The __call__ method is provided for convenience, allowing layers to be
    called as functions: output = layer(input)
    """

    def __init__(self) -> None:
        """Initialize the layer."""
        super().__init__()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Enable layer to be called as function: output = layer(input).

        Args:
            x: Input tensor of shape (N, ...).

        Returns:
            Output tensor from forward pass.
        """
        return self.forward(x)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Define the computation performed at every call (forward pass).

        Args:
            x: Input tensor of shape (N, ...).

        Returns:
            Output tensor of shape (N, ...).

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError("Layer subclasses must implement forward method")

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Define the backward pass for computing gradients.

        Args:
            grad: Gradient from subsequent layer of shape (batch_size, ...).

        Returns:
            Gradient with respect to input of this layer.

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError("Layer subclasses must implement backward method")
