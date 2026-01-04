from __future__ import annotations
from typing import Any
from numpy import ndarray


class Context:
    """
    Container for saving intermediate values during forward pass.

    Context acts as a communication channel between forward and backward
    passes of a Function. Values saved during forward (e.g., inputs,
    intermediate results) can be retrieved in backward to compute gradients.

    This design separates concerns:
    - Forward pass: computes output and saves necessary values
    - Backward pass: retrieves saved values to compute gradients

    Attributes:
        saved_tensors: Tuple of numpy arrays saved via save_for_backward().
        saved_shapes: Tuple of shape tuples for reference (optional usage).

    Examples:
        >>> class MyOp(Function):
        ...     @staticmethod
        ...     def forward(ctx, x, y):
        ...         ctx.save_for_backward(x, y)
        ...         return x * y
        ...
        ...     @staticmethod
        ...     def backward(ctx, grad_output):
        ...         x, y = ctx.saved_tensors
        ...         return grad_output * y, grad_output * x
    """

    def __init__(self):
        """Initializes an empty context."""
        self.saved_tensors: tuple[ndarray, ...] | Any = ()
        self.saved_shapes: tuple[tuple[int, ...], ...] = ()

    def save_for_backward(self, *args: ndarray) -> None:
        """
        Saves numpy arrays for use in the backward pass.

        Args:
            *args: Numpy arrays to save. Typically inputs or intermediate
                results needed to compute gradients.

        Examples:
            >>> ctx = Context()
            >>> x = np.array([1.0, 2.0])
            >>> y = np.array([3.0, 4.0])
            >>> ctx.save_for_backward(x, y)
            >>> print(ctx.saved_tensors)  # (array([1., 2.]), array([3., 4.]))
        """
        self.saved_tensors = args
