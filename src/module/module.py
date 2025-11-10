import numpy as np
from typing import Iterable


class Parameters:
    """A wrapper for a tensor that is considered a model parameter.

    This class holds the parameter data (e.g., weights or biases) and its
    corresponding gradient.
    """

    def __init__(self, data: np.ndarray):
        """Initializes the Parameters object.

        Args:
            data (np.ndarray): The initial data for the parameter.
        """
        self.data = data
        # Initialize gradient with the same shape as data, filled with zeros.
        self.grad = np.zeros_like(data, dtype=np.float32)
        self.name = None

    def zero_grad(self):
        """Resets the gradient of the parameter to zero."""
        self.grad.fill(0.0)


class Module:
    """Base class for all neural network modules.

    Your models should also subclass this class. Modules can contain other
    modules, allowing to nest them in a tree structure.
    """

    def __init__(self):
        """Initializes the module, setting the mode to training by default."""
        self._training = True

    def train(self):
        """Sets the module in training mode.

        This has an effect on certain modules like Dropout or BatchNorm.
        """
        self._training = True

    def eval(self):
        """Sets the module in evaluation mode.

        This has an effect on certain modules like Dropout or BatchNorm.
        """
        self._training = False

    def parameters(self) -> Iterable[Parameters]:
        """Returns an iterator over module parameters.

        This is typically overridden by subclasses that have parameters.

        Returns:
            Iterable[Parameters]: An iterable of the module's parameters.
        """
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
