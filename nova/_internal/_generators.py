from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Any, Type, Callable
from nova.utils import ensure_tensor


if TYPE_CHECKING:
    from nova.autograd.function import Function
    from nova import Tensor


def make_reverse_func(
    func: Type[Function],
) -> Callable[[Tensor, Tensor | Any], Tensor]:
    """
    Generates a reverse operation method wrapper (e.g., __radd__).

    Reverse operations are called when the left operand doesn't support
    the operation. For example, `5 + tensor` calls `tensor.__radd__(5)`.

    Args:
        func (Function): The Function class implementing the operation's forward/backward logic.

    Returns:
        A method that swaps operand order and applies the operation.

    Examples:
        >>> # Internal usage: binding __radd__ for Add operation
        >>> __radd__ = make_reverse_func(Add)
        >>> result = 5 + tensor  # Calls tensor.__radd__(5) -> Add.apply(5, tensor)
    """
    from nova import Tensor

    def method(self: Tensor, other: Tensor | Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = ensure_tensor(other)
        return func.apply(other, self)

    return method


def make_method(func: Type[Function]) -> Callable[[Tensor, ...], Tensor]:  # type: ignore
    """
    Generates a standard method wrapper for operations.

    Creates a simple forwarding method that passes all arguments directly
    to the Function's apply method. Used for regular named methods like
    `tensor.add(other)` or `tensor.pow(2)`.

    Args:
        func (Function): The Function class implementing the operation.

    Returns:
        A method that forwards arguments to the Function's apply method.

    Examples:
        >>> # Internal usage: binding regular methods
        >>> add_method = make_method(Pow)
        >>> result = tensor.pow(other)  # Calls Pow.apply(tensor, other)
    """

    def method(self: Tensor, *args, **kwargs) -> Tensor:
        return func.apply(self, *args, **kwargs)

    return method


def make_forward_func(
    func: Type[Function],
    raw: bool,
    is_unary: bool,
) -> Callable[[Tensor, ...], Tensor]:  # type: ignore
    """
    Generates a forward operation method wrapper (e.g., __add__, __mul__).

    This is the primary method generator for dunder methods like __add__.
    It handles argument validation, automatic type conversion, and supports
    both unary and binary operations.

    Args:
        func (Function): The Function class implementing the operation.
        raw (bool): If True, skips automatic Tensor conversion for arguments.
            Used for operations that need raw scalar values (e.g. __getitem__).
        is_unary (bool): If True, operation takes only self as input (e.g., __neg__).

    Returns:
        A method that validates inputs and applies the operation.

    Raises:
        TypeError: If a binary operation is called without arguments.

    Examples:
        >>> # Internal usage: binding __add__ for Add operation
        >>> __add__ = make_forward_func(Add, raw=False, is_unary=False)
        >>> result = tensor + 5  # Calls Add.apply(tensor, 5)

        >>> # Unary operation example
        >>> __neg__ = make_forward_func(Neg, raw=False, is_unary=True)
        >>> result = -tensor  # Calls Neg.apply(tensor)
    """
    from nova import Tensor

    def method(self: Tensor, *args, **kwargs) -> Tensor:
        # Handle unary operations (no additional arguments needed)
        if is_unary:
            return func.apply(self)

        # Validate that binary operations have an argument
        if not args and not kwargs:
            raise TypeError(f"Operation {func.__name__} requires an 'other' argument.")

        other = args[0] if args else kwargs.get("other")

        # Convert to Tensor unless raw mode is enabled
        if not raw and not isinstance(other, Tensor):
            other = ensure_tensor(other)

        return func.apply(self, other)

    return method


def make_inplace_func(
    func: Type[Function], raw: bool, op_name: str, is_unary: bool
) -> Callable[[Tensor, ...], Tensor]:  # type: ignore
    """
    Generates an in-place operation method wrapper (e.g., add_, __iadd__).

    In-place operations modify the tensor's data directly without creating
    a new tensor. They are more memory-efficient but incompatible with
    autograd on tensors that require gradients.

    The generated method:
    1. Validates that the tensor doesn't require gradients
    2. Applies the operation
    3. Handles dtype casting if necessary
    4. Copies result back into the original tensor's data buffer
    5. Returns self for method chaining

    Args:
        func (Function): The Function class implementing the operation.
        raw (bool): If True, skips automatic Tensor conversion for arguments.
        op_name (str): Name of the operation for error messages.
        is_unary (bool): If True, operation takes only self as input.

    Returns:
        A method that performs the operation in-place and returns self.

    Raises:
        RuntimeError: If called on a tensor that requires gradients.
        TypeError: If a binary operation is called without arguments.

    Examples:
        >>> # Internal usage: binding add_ for Add operation
        >>> add_ = make_inplace_func(Add, raw=False, op_name='add', is_unary=False)
        >>> x = nova.tensor([1.0, 2.0], requires_grad=False)
        >>> x.add_(5)  # Modifies x in-place
        >>> print(x)  # tensor([6.0, 7.0])

        >>> # Error case: tensor with gradients
        >>> y = nova.tensor([1.0, 2.0], requires_grad=True)
        >>> y.add_(5)  # RuntimeError: Cannot perform inplace operation
    """
    from nova import Tensor

    def inplace_method(self: Tensor, *args, **kwargs) -> Tensor:
        # Prevent in-place ops on tensors in the computation graph
        if self.requires_grad:
            raise RuntimeError(
                f"Cannot perform inplace operation '{op_name}_' on a tensor "
                f"that requires gradients. Use the out-of-place version instead."
            )

        # Handle unary operations
        if is_unary:
            result = func.apply(self).data
        else:
            # Validate binary operation arguments
            if not args and not kwargs:
                raise TypeError(
                    f"Operation {func.__name__} requires an 'other' argument."
                )

            other = args[0] if args else kwargs.get("other")
            remaining_args = args[1:] if len(args) > 1 else ()

            # Convert to Tensor unless raw mode is enabled
            if not raw and not isinstance(other, Tensor):
                other = ensure_tensor(other)

            result = func.apply(self, other, *remaining_args, **kwargs).data

        # Preserve original dtype through casting if needed
        if result.dtype != self.data.dtype:
            result = result.astype(self.data.dtype)

        # Copy result into original tensor's data buffer
        np.copyto(dst=self.data, src=result)

        return self

    return inplace_method
