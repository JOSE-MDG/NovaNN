from __future__ import annotations
from nova.utils import ensure_tensor

from typing import (
    TYPE_CHECKING,
    Any,
    Type,
    overload,
    Union,
)

if TYPE_CHECKING:
    from nova.autograd.function import Function
    from nova import Tensor
    from nova._typing import (
        UnaryMethod,
        BinaryMethod,
        ReverseBinaryMethod,
        VariadicMethod,
        InplaceUnaryMethod,
        InplaceBinaryMethod,
        T,
    )


def make_reverse_func(
    func: Type[Function],
) -> ReverseBinaryMethod:
    """
    Generates a reverse operation method wrapper (e.g., __radd__).

    Reverse operations are called when the left operand doesn't support
    the operation. For example, `5 + tensor` calls `tensor.__radd__(5)`.

    Args:
        func: The Function class implementing the operation's forward/backward logic.

    Returns:
        A reverse binary method that swaps operand order and applies the operation.

    Type Safety:
        - Input: `(self: Tensor, other: Tensor | Any) -> Tensor`
        - Automatically converts non-Tensor arguments to Tensors
        - Returns a new Tensor (non-mutating)

    Examples:
        >>> # Internal usage: binding __radd__ for Add operation
        >>> __radd__ = make_reverse_func(Add)
        >>> result = 5 + tensor  # Calls tensor.__radd__(5) -> Add.apply(5, tensor)
    """
    from nova import Tensor

    def method(self: Tensor, other: Union[Tensor, Any], /) -> Tensor:
        if not isinstance(other, Tensor):
            other = ensure_tensor(other)
        return func.apply(other, self)

    return method


def make_method(func: Type[Function]) -> VariadicMethod:
    """
    Generates a standard method wrapper for operations.

    Creates a simple forwarding method that passes all arguments directly
    to the Function's apply method. Used for regular named methods like
    `tensor.add(other)` or `tensor.pow(2)`.

    Args:
        func: The Function class implementing the operation.

    Returns:
        A variadic method that forwards all arguments to Function.apply.

    Type Safety:
        - Input: `(self: Tensor, *args, **kwargs) -> Tensor`
        - Flexible signature for different operation types
        - Returns a new Tensor (non-mutating)

    Examples:
        >>> # Internal usage: binding regular methods
        >>> pow_method = make_method(Pow)
        >>> result = tensor.pow(2)  # Calls Pow.apply(tensor, 2)

        >>> sum_method = make_method(Sum)
        >>> result = tensor.sum(dim=0, keepdims=True)  # Calls Sum.apply(tensor, dim=0, keepdims=True)
    """

    def method(self: Tensor, /, *args: Any, **kwargs: Any) -> Tensor:
        return func.apply(self, *args, **kwargs)

    return method


@overload
def make_forward_func(
    func: Type[Function],
    raw: bool,
    is_unary: bool = False,
) -> UnaryMethod: ...


@overload
def make_forward_func(
    func: Type[Function],
    raw: bool,
    is_unary: bool = False,
) -> BinaryMethod: ...


def make_forward_func(
    func: Type[Function],
    raw: bool,
    is_unary: bool,
) -> Union[UnaryMethod, BinaryMethod]:
    """
    Generates a forward operation method wrapper (e.g., __add__, __mul__).

    This is the primary method generator for dunder methods like __add__.
    It handles argument validation, automatic type conversion, and supports
    both unary and binary operations.

    Args:
        func: The Function class implementing the operation.
        raw: If True, skips automatic Tensor conversion for arguments.
            Used for operations that need raw scalar values (e.g. __getitem__).
        is_unary: If True, operation takes only self as input (e.g., __neg__).

    Returns:
        A unary or binary method depending on the is_unary flag.

    Type Safety:
        - Unary: `(self: Tensor) -> Tensor`
        - Binary: `(self: Tensor, other: Tensor | Any) -> Tensor`
        - Auto-converts non-Tensor arguments unless raw=True
        - Returns a new Tensor (non-mutating)

    Raises:
        TypeError: If a binary operation is called without arguments.

    Examples:
        >>> # Unary operation
        >>> __neg__ = make_forward_func(Neg, raw=False, is_unary=True)
        >>> result = -tensor  # Calls Neg.apply(tensor)

        >>> # Binary operation with auto-conversion
        >>> __add__ = make_forward_func(Add, raw=False, is_unary=False)
        >>> result = tensor + 5  # Calls Add.apply(tensor, ensure_tensor(5))

        >>> # Binary operation with raw args (no conversion)
        >>> __getitem__ = make_forward_func(GetItem, raw=True, is_unary=False)
        >>> result = tensor[0]  # Calls GetItem.apply(tensor, 0) - index stays int
    """
    from nova import Tensor

    if is_unary:

        def unary_method(self: Tensor, /) -> Tensor:
            return func.apply(self)

        return unary_method

    else:

        def binary_method(self: Tensor, other: Union[Tensor, Any], /) -> Tensor:
            # Convert to Tensor unless raw mode is enabled
            if not raw and not isinstance(other, Tensor):
                other = ensure_tensor(other)

            return func.apply(self, other)

        return binary_method


@overload
def make_inplace_func(
    func: Type[Function],
    raw: bool,
    op_name: str,
    is_unary: bool = True,
) -> InplaceUnaryMethod: ...


@overload
def make_inplace_func(
    func: Type[Function],
    raw: bool,
    op_name: str,
    is_unary: bool = False,
) -> InplaceBinaryMethod: ...


def make_inplace_func(
    func: Type[Function],
    raw: bool,
    op_name: str,
    is_unary: bool,
) -> Union[InplaceUnaryMethod, InplaceBinaryMethod]:
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
        func: The Function class implementing the operation.
        raw: If True, skips automatic Tensor conversion for arguments.
        op_name: Name of the operation for error messages.
        is_unary: If True, operation takes only self as input.

    Returns:
        An in-place unary or binary method that modifies self and returns self.

    Type Safety:
        - Unary: `(self: T) -> T` where T is bound to Tensor
        - Binary: `(self: T, other: Tensor | Any) -> T`
        - Returns the same instance (mutating operation)
        - Generic T ensures type checkers know it's the same object

    Raises:
        RuntimeError: If called on a tensor that requires gradients.
        TypeError: If a binary operation is called without arguments.

    Examples:
        >>> # Unary in-place
        >>> abs_ = make_inplace_func(Abs, raw=False, op_name='abs', is_unary=True)
        >>> x = nova.tensor([-1.0, 2.0], requires_grad=False)
        >>> x.abs_()  # Modifies x in-place, returns x
        >>> print(x)  # tensor([1.0, 2.0])

        >>> # Binary in-place with chaining
        >>> add_ = make_inplace_func(Add, raw=False, op_name='add', is_unary=False)
        >>> x = nova.tensor([1.0, 2.0], requires_grad=False)
        >>> x.add_(5).mul_(2)  # Chain in-place operations
        >>> print(x)  # tensor([12.0, 14.0])

        >>> # Error case: tensor with gradients
        >>> y = nova.tensor([1.0, 2.0], requires_grad=True)
        >>> y.add_(5)  # RuntimeError: Cannot perform inplace operation
    """
    from nova import Tensor

    if is_unary:

        def inplace_unary_method(self: T, /) -> T:
            # Prevent in-place ops on tensors in the computation graph
            if self.requires_grad:
                raise RuntimeError(
                    f"Cannot perform inplace operation '{op_name}_' on a tensor "
                    f"that requires gradients. Use the out-of-place version instead."
                )

            func.apply(self, _out=self.data)

            return self

        return inplace_unary_method

    else:

        def inplace_binary_method(self: T, *args: Union[Tensor, Any]) -> T:
            # Prevent in-place ops on tensors in the computation graph
            if self.requires_grad:
                raise RuntimeError(
                    f"Cannot perform inplace operation '{op_name}_' on a tensor "
                    f"that requires gradients. Use the out-of-place version instead."
                )

            # Convert to Tensor unless raw mode is enabled
            if not raw and len(args) == 1 and not isinstance(args[0], Tensor):
                args = (ensure_tensor(args[0]),)

            func.apply(self, *args, _out=self.data)

            return self

        return inplace_binary_method
