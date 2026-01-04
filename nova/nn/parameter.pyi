from typing_extensions import TypeIs
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional
from nova import Tensor

if TYPE_CHECKING:
    from nova._typing import Dtype, Size

class Parameter(Tensor):
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...

class Buffer(Tensor):
    persistent: bool
    def __init__(
        self,
        data: Tensor = ...,
        *,
        persistent: bool = ...,
    ) -> None: ...

def is_lazy(
    param: Tensor,
) -> TypeIs[UninitializedParameter | UninitializedBuffer]: ...

class UninitializedParameter(UninitializedTensorMixin, Parameter):
    def __init__(self, requires_grad: bool = ...) -> None: ...
    def materialize(self, shape: Size, dtype: Optional[Dtype] = None) -> Parameter: ...

class UninitializedBuffer(UninitializedTensorMixin, Buffer):
    def __init__(self, persistent: bool = ...) -> None: ...
    def materialize(self, shape: Size, dtype: Optional[Dtype] = None) -> Buffer: ...

class UninitializedTensorMixin:
    @abstractmethod
    def materialize(self, shape: Size, dtype: Optional[Dtype] = None) -> Tensor: ...
    @property
    def shape(self): ...
    def __repr__(self) -> str: ...
