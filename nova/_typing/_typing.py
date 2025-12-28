from __future__ import annotations
from nova import dtypes
from numpy import ndarray
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Type, Union

if TYPE_CHECKING:
    from nova import Tensor
    from nova._interfaces._optimizer import Optimizer
    from nova._interfaces._base_tensor import TensorBase
    from nova._interfaces._lr_scheduler import _LRScheduler
    from nova.nn.modules.module import Module, Parameter, Buffer

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
type StepHook = Callable[[Optimizer], None]
type Hooks = Hook | StepHook
type HooksList = list[Hook] | list[StepHook] | list[Hooks]

type Gradients = tuple[ndarray | None, ...]
type Dim = tuple[int, ...] | int
type Closure = Optional[Callable[[], Optional[float]]]
type Modules = Union[
    Tensor, Optimizer, TensorBase, _LRScheduler, Parameter, Buffer, Module
]

type ModuleTypes = Type[
    Union[Tensor, Optimizer, TensorBase, _LRScheduler, Parameter, Buffer, Module]
]

type KernelSize = int | tuple[int, int] | tuple[int, int, int]
type Stride = int | tuple[int, int] | tuple[int, int, int]
type Padding = (int | tuple[int, int] | tuple[int, int, int] | Literal["valid", "same"])
type Defaults = dict[str, Any]
type Group = dict[str, Union[list[Parameter], Any]]
type ParamGroups = list[Group]
type State = dict[Parameter, dict[str, Any]]

type OptimizerStateDict = dict[
    Literal["state", "param_groups"], Union[State, list[dict[str, Any]]]
]

type SchedulerStateDict = dict[Literal["base_lrs", "last_epoch"], list[float] | int]
type YAMLFile = list[dict[str, dict[str, Any]]]
