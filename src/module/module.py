import numpy as np
from typing import Optional
from src._typing import ListOfParameters

"""
Module and parameter container utilities.

Provides:
- Parameters: Lightweight wrapper for trainable arrays and their gradients
- Module: Base class for all network modules with parameter discovery
"""


class Parameters:
    """Wrapper for tensors that are considered model parameters.

    Holds parameter data (weights/biases) and corresponding gradients.

    Args:
        data: Initial parameter array.

    Attributes:
        data: Parameter values.
        grad: Accumulated gradient or None.
        name: Optional human-readable identifier.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = np.zeros_like(self.data)
        self.name: Optional[str] = None

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset the stored gradient.

        Args:
            set_to_none: If True, set grad to None. If False, zero the array
                (recreating it if it was None).
        """
        if set_to_none:
            self.grad = None
        else:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            else:
                self.grad.fill(0.0)


class Module:
    """Base class for all neural network modules.

    Modules may contain Parameters and other Modules. The default `parameters()`
    method returns an empty list - subclasses should override to expose their
    trainable parameters.

    Tracks training/eval mode for layers that change behavior (Dropout, BatchNorm).
    """

    def __init__(self) -> None:
        """Initialize module in training mode by default."""
        self._training: bool = True

    def train(self) -> None:
        """Set the module to training mode (affects certain submodules)."""
        self._training = True

    def eval(self) -> None:
        """Set the module to evaluation mode (affects certain submodules)."""
        self._training = False

    def parameters(self) -> ListOfParameters:
        """Return list of Parameters belonging to this module.

        Subclasses must override this method to expose trainable parameters.
        Example implementation:

        ```python
        def parameters(self):
            params = [self.weight]
            if self.bias is not None:
                params.append(self.bias)
            return params
        ```

        Returns:
            List of parameter objects. Empty list by default.
        """
        return []

    def zero_grad(self) -> None:
        """Zero or clear gradients for all parameters in this module."""
        for param in self.parameters():
            param.zero_grad()
