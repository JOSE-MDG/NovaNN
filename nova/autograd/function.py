from __future__ import annotations
import nova
from .engine import Context
from typing import Any, TYPE_CHECKING, Type
from numpy import ndarray
from abc import ABC, ABCMeta

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Gradients


class FunctionMeta(ABCMeta):
    def __repr__(cls) -> str:
        return f"{cls.__name__}Backward"


class Function(ABC, metaclass=FunctionMeta):
    @staticmethod
    def forward(ctx: Context, *args: Any, **kwargs) -> ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        raise NotImplementedError

    @classmethod
    def apply(cls: Type[Function], *args: Any, **kwargs: Any) -> Tensor:
        from nova import Tensor

        ctx: Context = Context()
        tensors_in_graph: list[Tensor] = []

        def process_arg(arg: Any) -> Any:
            if isinstance(arg, Tensor):
                tensors_in_graph.append(arg)
                return arg.data
            elif isinstance(arg, list):
                return [process_arg(a) for a in arg]
            elif isinstance(arg, tuple):
                return tuple(process_arg(a) for a in arg)
            elif isinstance(arg, dict):
                return {k: process_arg(v) for k, v in arg.items()}
            return arg

        raw_args = tuple(process_arg(a) for a in args)
        raw_kwargs = {k: process_arg(v) for k, v in kwargs.items()}

        output = cls.forward(ctx, *raw_args, **raw_kwargs)

        requires_grad = (
            any(t.requires_grad for t in tensors_in_graph) and nova.is_grad_enabled()
        )

        result = Tensor(
            output,
            requires_grad=requires_grad,
            dtype=output.dtype,
            grad_fn=cls if requires_grad else None,
            copy=False,
        )

        if requires_grad:
            result._inputs = tensors_in_graph
            result._ctx = ctx
            if tensors_in_graph:
                result.rank = max(t.rank for t in tensors_in_graph) + 1

        return result
