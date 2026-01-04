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
    from nova.nn import Module, Parameter, Buffer
    from nova._interfaces._optimizer import Optimizer
    from nova._interfaces._base_tensor import TensorBase
    from nova._interfaces._lr_scheduler import _LRScheduler


# Core Tensor Types


type Size = tuple[int, ...]
"""Shape representation as a tuple of dimension sizes."""

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
"""Supported data types for tensor elements."""

type Dim = tuple[int, ...] | int
"""Dimension specification - either single dimension index or tuple of indices."""

type TensorOrArray = Union[Tensor, ndarray, list[Tensor], tuple[Tensor, ...]]
"""Flexible type for tensor-like inputs."""

type Inputs = Union[TensorOrArray, int, float, Any]
"""General input type for operations (tensors, scalars, or arrays)."""

type TensorsOrArrays = (
    tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
    | tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray], tuple[ndarray, ndarray]]
)
"""Type for training data batches (typically features, targets, and optionally weights)."""


# Autograd Types


type Gradients = tuple[ndarray | None, ...]
"""
Tuple of gradients returned by Function.backward().
None indicates an input doesn't require gradients.
"""

type Hook = Callable[[ndarray], Optional[ndarray]]
"""Backward hook that receives and optionally modifies gradients."""

type StepHook = Callable[[Optimizer], None]
"""Optimizer step hook called after parameter updates."""

type Hooks = Hook | StepHook
"""Union of all hook types."""

type HooksList = list[Hook] | list[StepHook] | list[Hooks]
"""List of hooks of any type."""

type Closure = Optional[Callable[[], Optional[float]]]
"""Optional closure function for optimizers that reevaluate the model."""


# Module and Parameter Types


type Modules = Union[
    Tensor, Optimizer, TensorBase, _LRScheduler, Parameter, Buffer, Module
]
"""Union of all module-like objects in Nova."""

type ModuleTypes = Type[
    Union[Tensor, Optimizer, TensorBase, _LRScheduler, Parameter, Buffer, Module]
]
"""Type objects for module-like classes."""


# Convolution and Pooling Types


type KernelSize = int | tuple[int, int] | tuple[int, int, int]
"""Kernel size for convolution/pooling - supports 1D, 2D, and 3D."""

type Stride = Optional[int | tuple[int, int] | tuple[int, int, int]]
"""Stride for convolution/pooling operations."""

type Dilation = int | tuple[int, int] | tuple[int, int, int]
"""Dilation rate for dilated convolutions."""

type Padding = int | tuple[int, int] | tuple[int, int, int] | Literal["valid", "same"]
"""Padding specification - explicit values or 'valid'/'same' modes."""

type PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]
"""Padding mode for how boundary values are filled."""


# Optimizer Types


type Defaults = dict[str, Any]
"""Default hyperparameters for optimizer."""

type Group = dict[str, Union[list[Parameter], Any]]
"""Parameter group with associated hyperparameters."""

type ParamGroups = list[Group]
"""List of parameter groups in an optimizer."""

type State = dict[Parameter, dict[str, Any]]
"""Optimizer state dictionary mapping parameters to their state."""

type OptimizerStateDict = dict[
    Literal["state", "param_groups"], Union[State, list[dict[str, Any]]]
]
"""Complete optimizer state for serialization."""


# Scheduler Types


type SchedulerStateDict = dict[Literal["base_lrs", "last_epoch"], list[float] | int]
"""Learning rate scheduler state for serialization."""


# Loss Function Types


type LossReducton = Literal["none", "mean", "sum"] | Literal[
    "none", "mean", "sum", "batchmean"
]
"""Reduction mode for loss functions."""


# Operation Registry Types


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
"""Registered operation names for YAML binding."""


# YAML Configuration Types


class InplaceInfo(TypedDict, total=False):
    """Configuration for in-place operation variants."""

    method: str
    """In-place method name (e.g., 'add_')."""

    dunder: str
    """In-place dunder method name (e.g., '__iadd__')."""


class TensorInfo(TypedDict, total=False):
    """Configuration for binding operations to Tensor class."""

    dunder: str
    """Dunder method name (e.g., '__add__')."""

    reverse: str
    """Reverse dunder method name (e.g., '__radd__')."""

    method: str
    """Regular method name (e.g., 'add')."""

    inplace: Union[InplaceInfo, Literal[False]]
    """In-place variant configuration or False if not supported."""


class OperationInfo(TypedDict, total=False):
    """Complete operation definition for YAML configuration."""

    name: OperationName
    """Registered operation name."""

    is_unary: bool
    """True if operation takes only one tensor input."""

    raw_args: bool
    """True if arguments should not be auto-converted to tensors."""

    tensor: TensorInfo
    """Tensor method binding configuration."""


class YAMLFile(TypedDict):
    """Root structure of native_functions.yaml."""

    ops: List[OperationInfo]
    """List of all operation definitions."""
