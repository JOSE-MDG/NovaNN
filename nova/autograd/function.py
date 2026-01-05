from __future__ import annotations
import nova
import numpy as np
from .engine import Context
from .utils import ArgumentProcessor, determine_base_dtype
from typing import Any, TYPE_CHECKING, Type
from abc import ABC, ABCMeta

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Gradients


class FunctionMeta(ABCMeta):
    def __repr__(cls) -> str:
        return f"<{cls.__name__}Backward>"


class Function(ABC, metaclass=FunctionMeta):
    @staticmethod
    def forward(ctx: Context, *args: Any, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Gradients:
        raise NotImplementedError

    @classmethod
    def apply(cls: Type[Function], *args: Any, **kwargs: Any) -> Tensor:
        from nova import Tensor

        ctx = Context()

        base_dtype = determine_base_dtype(args)
        processor = ArgumentProcessor(base_dtype)

        raw_args, raw_kwargs = processor.process_args(args, kwargs)
        tensors_in_graph = processor.get_tracked_tensors()

        output = cls.forward(ctx, *raw_args, **raw_kwargs)

        if not isinstance(output, np.ndarray):
            output = np.array(output)

        output_dtype = output.dtype

        if (
            not np.issubdtype(output_dtype, np.bool_)
            and not np.issubdtype(output_dtype, np.integer)
            and not np.issubdtype(output_dtype, np.complexfloating)
        ):
            output = output.astype(base_dtype, copy=False)
            output_dtype = base_dtype

        requires_grad = (
            any(t.requires_grad for t in tensors_in_graph) and nova.is_grad_enabled()
        )

        result = Tensor(
            output,
            requires_grad=requires_grad,
            dtype=output_dtype,
            grad_fn=cls if requires_grad else None,
            copy=False,
        )

        if requires_grad:
            result._inputs = tensors_in_graph
            result._ctx = ctx

        return result
