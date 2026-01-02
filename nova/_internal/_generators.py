from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Any, Type, Callable
from nova.utils import ensure_tensor


if TYPE_CHECKING:
    from nova.autograd.function import Function
    from nova import Tensor


def make_reverse_func(func: Type[Function]) -> Callable[[Tensor, Tensor | Any], Tensor]:
    from nova import Tensor

    def method(self: Tensor, other: Tensor | Any) -> Tensor:
        if not isinstance(other, Tensor):
            other = ensure_tensor(other)
        return func.apply(other, self)

    return method


def make_method(func: Type[Function]) -> Callable[[Tensor, Any], Tensor]:
    def method(self: Tensor, *args, **kwargs) -> Tensor:
        return func.apply(self, *args, **kwargs)

    return method


def make_forward_func(
    func: Type[Function],
    raw: bool,
    is_unary: bool,
) -> Callable[[Tensor, Any], Tensor]:
    from nova import Tensor

    def method(self: Tensor, *args, **kwargs) -> Tensor:
        if is_unary:
            return func.apply(self)

        if not args and not kwargs:
            raise TypeError(f"Operation {func.__name__} requires an 'other' argument.")

        other = args[0] if args else kwargs.get("other")

        if not raw and not isinstance(other, Tensor):
            other = ensure_tensor(other)

        return func.apply(self, other)

    return method


def make_inplace_func(
    func: Type[Function], raw: bool, op_name: str, is_unary: bool
) -> Callable[[Tensor, Any], Tensor]:
    from nova import Tensor

    def inplace_method(self: Tensor, *args, **kwargs) -> Tensor:
        if self.requires_grad:
            raise RuntimeError(
                f"Cannot perform inplace operation '{op_name}_' on a tensor "
                f"that requires gradients. Use the out-of-place version instead."
            )

        if is_unary:
            return func.apply(self)

        if not args and not kwargs:
            raise TypeError(f"Operation {func.__name__} requires an 'other' argument.")

        other = args[0] if args else kwargs.get("other")
        remainig_args = args[1:] if len(args) > 1 else ()

        if not raw and not isinstance(other, Tensor):
            other = ensure_tensor(other)

        result = func.apply(self, other, *remainig_args, *kwargs).data

        if result.dtype != self.data.dtype:
            result = result.astype(self.data.dtype)

        np.copyto(dst=self.data, src=result)

        return self

    return inplace_method
