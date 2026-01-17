from __future__ import annotations
from typing import Any, Type, TypeVar
from abc import ABCMeta
from numpy import ndarray
from nova import Tensor
from nova._typing import Gradients
from .engine import Context

TFunction = TypeVar("TFunction", bound=Function)

class FunctionMeta(ABCMeta):
    def __repr__(cls) -> str: ...

class Function(metaclass=FunctionMeta):
    @staticmethod
    def forward(ctx: Context, *args: Any, **kwargs: Any) -> ndarray: ...
    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients: ...
    @classmethod
    def apply(cls: Type[TFunction], *args: Any, **kwargs: Any) -> Tensor: ...
