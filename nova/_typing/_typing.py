from __future__ import annotations
from nova import dtypes
from numpy import ndarray
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Type,
    Union,
)

if TYPE_CHECKING:
    from nova._tensor import Tensor
    from nova._interfaces._optimizer import Optimizer
    from nova._interfaces._base_tensor import TensorBase
    from nova._interfaces._lr_scheduler import _LRScheduler
    from nova.nn import Parameter, Buffer, Module

type Size = tuple[int, ...]

type Dtype = Union[
    dtypes.uint8,
    dtypes.int8,
    dtypes.short,
    dtypes.int,
    dtypes.long,
    dtypes.float32,
    dtypes.double,
    dtypes.float128,
    dtypes.half,
    dtypes.bool,  # type: ignore
]

type TensorOrArray = Union[Tensor, ndarray, list[Tensor], tuple[Tensor, ...]]
type Inputs = Union[TensorOrArray, int, float, Any]
type Hook = Callable[[ndarray], Optional[ndarray]]
type StepHook = Callable[[Type[Optimizer]], None]
type Gradients = tuple[ndarray | None, ...]
type Dim = tuple[int, ...] | int
type Closure = Optional[Callable[[], Optional[float]]]
type Modules = Union[
    Tensor, Optimizer, TensorBase, _LRScheduler, Parameter, Buffer, Module
]  # type: ignore

# Conv aliases
type KernelSize = int | tuple[int, int] | tuple[int, int, int]
type Stride = int | tuple[int, int] | tuple[int, int, int]
type Padding = (int | tuple[int, int] | tuple[int, int, int] | Literal["valid", "same"])
"""
{ param_group
    'params':list[Paramater],
    'lr':float,
    'momentum':float
}
{ state
    Parameter:dict[str, float | ndarray | int]
}
"""
type Defaults = dict[str, float | bool | tuple[float, float] | int]
type ParamGroups = list[dict[str, list[Parameter] | Defaults]]
type State = dict[Parameter, dict[str, float | ndarray | int]]
type OptimizerStateDict = dict[str, ParamGroups | State]
