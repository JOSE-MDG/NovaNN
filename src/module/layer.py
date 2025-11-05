import numpy as np
from src.module.module import Module


class Layer(Module):
    """Abstract base class for all layers in the neural network.

    This class inherits from Module and defines the interface that all
    neural network layers must implement, including the forward and backward
    passes.
    """

    def __init__(self):
        """Initializes the layer.

        Sets a `_have_params` flag to False by default. Subclasses with
        trainable parameters should set this flag to True.
        """
        super().__init__()
        self._have_params = False

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

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Defines the backward pass for computing gradients.

        This method must be overridden by all subclasses.

        Args:
            grad (np.ndarray): The gradient from the subsequent layer.

        Returns:
            np.ndarray: The gradient with respect to the input of this layer.
        """
        raise NotImplementedError
