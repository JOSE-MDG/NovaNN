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

type Size = tuple[int, ...]

type Dtype = (
    dtypes.uint8
    | dtypes.int8
    | dtypes.short
    | dtypes.int
    | dtypes.long
    | dtypes.half
    | dtypes.float32
    | dtypes.double
    | dtypes.float128
    | dtypes.bool
)

type TensorOrArray = Tensor | ndarray | list[Tensor] | tuple[Tensor, ...]
type Inputs = TensorOrArray | int | float | Any
type Hook = (Callable[[ndarray], Optional[ndarray]] | Callable[[Type[Optimizer]], Any])
type Gradients = tuple[ndarray | None, ...]
type Dim = tuple[int, ...] | int
type Closure = Optional[Callable[[], Optional[float]]]

# Conv aliases
type KernelSize = int | tuple[int, int] | tuple[int, int, int]
type Stride = int | tuple[int, int] | tuple[int, int, int]
type Padding = (int | tuple[int, int] | tuple[int, int, int] | Literal["valid", "same"])
