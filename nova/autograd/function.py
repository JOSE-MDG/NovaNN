from __future__ import annotations
import nova
from .engine import Context
from typing import Any, TYPE_CHECKING, Type
from nova._typing import Inputs
from numpy import ndarray
from abc import ABC, ABCMeta

if TYPE_CHECKING:
    from nova import Tensor


class FunctionMeta(ABCMeta):
    def __repr__(cls) -> str:
        return f"{cls.__name__}Backward"


class Function(ABC, metaclass=FunctionMeta):
    @staticmethod
    def forward(ctx: Context, *args: Any) -> ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> tuple[ndarray | None, ...]:
        raise NotImplementedError

    @classmethod
    def apply(cls: Type[Function], *args: Inputs, **kwargs) -> Tensor:
        from nova import Tensor

        ctx: Context = Context()
        raw_inputs: list[ndarray | Any] = []
        tensors: list[Tensor] = []

        for arg in args:
            if isinstance(arg, Tensor):
                raw_inputs.append(arg.data)
                tensors.append(arg)
            elif (
                isinstance(arg, (list, tuple))
                and len(arg) > 0
                and isinstance(arg[0], Tensor)
            ):
                raw_inputs.append([t.data for t in arg])
                tensors.extend(arg)
            else:
                raw_inputs.append(arg)

        output = cls.forward(ctx, *raw_inputs, **kwargs)

        requires_grad = any(t.requires_grad for t in tensors) and nova.is_grad_enabled()

        result = Tensor(
            output,
            requires_grad=requires_grad,
            dtype=output.dtype,
            grad_fn=cls if requires_grad else None,
            copy=False,
        )

        if requires_grad:
            result._inputs = tensors
            result._ctx = ctx

        return result
