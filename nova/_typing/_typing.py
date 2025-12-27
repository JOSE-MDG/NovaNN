from __future__ import annotations
from nova import dtypes
from numpy import ndarray
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, TypeAlias, Union

if TYPE_CHECKING:
    from nova._tensor import Tensor
    from nova._interfaces._optimizer import Optimizer

Size = tuple[int, ...]

Dtype = (
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

TensorOrArray: TypeAlias = Union[Tensor, ndarray, list[Tensor], tuple[Tensor, ...]]
Inputs: TypeAlias = TensorOrArray | int | float | Any
Hook: TypeAlias = (
    Callable[[ndarray], Optional[ndarray]] | Callable[[Type[Optimizer]], Any]
)
Gradients: TypeAlias = tuple[ndarray | None, ...]
Dim: TypeAlias = tuple[int, ...] | int
Closure: TypeAlias = Optional[Callable[[], Optional[float]]]
