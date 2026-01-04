from __future__ import annotations
from nova import dtypes
from numpy import ndarray
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Type,
    Union,
    TypedDict,
)

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

type TensorsOrArrays = tuple[
    tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]
] | tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray], tuple[ndarray, ndarray]]
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
type Stride = Optional[int | tuple[int, int] | tuple[int, int, int]]
type Dilation = int | tuple[int, int] | tuple[int, int, int]
type Padding = (int | tuple[int, int] | tuple[int, int, int] | Literal["valid", "same"])
type PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]
type Defaults = dict[str, Any]
type Group = dict[str, Union[list[Parameter], Any]]
type ParamGroups = list[Group]
type State = dict[Parameter, dict[str, Any]]

type OptimizerStateDict = dict[
    Literal["state", "param_groups"], Union[State, list[dict[str, Any]]]
]

type SchedulerStateDict = dict[Literal["base_lrs", "last_epoch"], list[float] | int]
type LossReducton = Literal["none", "mean", "sum"] | Literal[
    "none", "mean", "sum", "batchmean"
]

type OperationName = Literal[
    "relu",
    "leaky_relu",
    "prelu",
    "gelu",
    "sigmoid",
    "abs",
    "add",
    "arccos",
    "arcsin",
    "arctan",
    "ceil",
    "clamp",
    "cos",
    "cot",
    "det",
    "div",
    "divint",
    "dot",
    "exp",
    "extend",
    "floor",
    "getitem",
    "inv",
    "log",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "mod",
    "mul",
    "neg",
    "norm",
    "pad",
    "permute",
    "pow",
    "repeat",
    "reshape",
    "sec",
    "setitem",
    "sign",
    "sin",
    "split",
    "sqrt",
    "squeeze",
    "stride_tricks",
    "sub",
    "sum",
    "tan",
    "tanh",
    "tile",
    "trace",
    "unsqueeze",
    "view",
    "clone",
]


class InplaceInfo(TypedDict, total=False):
    method: str
    dunder: str


class TensorInfo(TypedDict, total=False):
    dunder: str
    reverse: str
    method: str
    inplace: Union[InplaceInfo, Literal[False]]


class OperationInfo(TypedDict, total=False):
    name: OperationName
    is_unary: bool
    raw_args: bool
    tensor: TensorInfo


class YAMLFile(TypedDict):

    ops: List[OperationInfo]
