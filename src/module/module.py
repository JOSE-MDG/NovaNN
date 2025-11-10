import numpy as np
from typing import Iterable, Iterator, Optional

"""
Module and parameter container utilities.

Provides:
- Parameters: lightweight wrapper for trainable arrays and their gradients.
- Module: base class for all network modules with a default parameters
  discovery implementation that walks attributes to yield Parameters
  (and recurses into nested Modules).
Docstrings follow the Google style.
"""


class Parameters:
    """A wrapper for a tensor that is considered a model parameter.

    This class holds the parameter data (e.g., weights or biases) and its
    corresponding gradient.

    Args:
        data: Initial parameter array.

    Attributes:
        data (np.ndarray): The parameter values.
        grad (Optional[np.ndarray]): The accumulated gradient or None.
        name (Optional[str]): Optional human-readable name assigned later.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data: np.ndarray = data
        # Initialize gradient with the same shape/dtype as the parameter data.
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

    Modules may contain Parameters and other Modules as attributes. The default
    `parameters()` implementation inspects instance attributes and yields
    found Parameters (recursing into nested Modules). Subclasses can override
    `parameters()` for custom behaviour.

    The Module tracks a training/eval mode flag which some layers use to
    change behaviour (e.g., Dropout, BatchNorm).
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

    def parameters(self) -> Iterable[Parameters]:
        """Yield Parameters belonging to this module.

        Default implementation inspects attributes of the instance and yields:
          - direct Parameters attributes,
          - Parameters contained in lists/tuples,
          - Parameters from nested Module attributes (recursing).

        Returns:
            An iterable (generator) of Parameters objects.
        """
        def _iter_params() -> Iterator[Parameters]:
            for attr in vars(self).values():
                if isinstance(attr, Parameters):
                    yield attr
                elif isinstance(attr, Module):
                    yield from attr.parameters()
                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, Parameters):
                            yield item
                        elif isinstance(item, Module):
                            yield from item.parameters()

        return _iter_params()

    def zero_grad(self) -> None:
        """Zero or clear gradients for all parameters in this module."""
        for p in self.parameters():
            p.zero_grad()
