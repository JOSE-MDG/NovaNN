"""
Layer base class.

Defines the interface every layer must implement (forward / backward)
and inherits from Module. Using an abstract base class makes it explicit
that Layer should not be instantiated directly.
"""

import numpy as np
from abc import ABC, abstractmethod
from src.module.module import Module


class Layer(Module, ABC):
    """Abstract base class for all layers in the neural network.

    This class inherits from Module and defines the interface that all
    neural network layers must implement, including the forward and backward
    passes.

    Subclasses must implement `forward` and `backward`.
    """

    def __init__(self) -> None:
        """Initializes the layer."""
        super().__init__()

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the computation performed at every call (forward pass).

        This method must be overridden by all subclasses.

        Args:
            x (np.ndarray): The input tensor.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Returns:
            np.ndarray: The output tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Defines the backward pass for computing gradients.

        This method must be overridden by all subclasses.

        Args:
            grad (np.ndarray): The gradient from the subsequent layer.

        Returns:
            np.ndarray: The gradient with respect to the input of this layer.
        """
        raise NotImplementedError
