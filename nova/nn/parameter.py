from __future__ import annotations
import nova
from abc import abstractmethod
from typing import Any, TYPE_CHECKING, Optional
from nova import Tensor
from nova.utils import registry_class

if TYPE_CHECKING:
    from nova._typing import Size, Dtype


@registry_class
class Parameter(Tensor):

    __slots__ = ["is_bn_param"]

    def __init__(
        self, data: Any, requires_grad: bool = True, dtype: Optional[Dtype] = None
    ) -> None:
        super().__init__(
            data=data, requires_grad=requires_grad, dtype=dtype, copy=False
        )

        self.is_bn_param: bool = False


@registry_class
class Buffer(Tensor):

    __slots__ = ["persistent"]

    def __init__(
        self, data: Any, *, dtype: Optional[Dtype] = None, persistent: bool = True
    ):
        super().__init__(data=data, dtype=dtype, requires_grad=False)
        self.persistent = persistent


@registry_class
class UninitializedTensorMixin:
    """Common base for all uninitialized tensors"""

    @abstractmethod
    def materialize(self, shape: Size, dtype: Optional[Dtype] = None):
        raise NotImplementedError

    @property
    def shape(self):
        raise RuntimeError(
            "Can't access shape of uninitialized parameter/buffer. "
            "Call forward() first."
        )


@registry_class
class UninitializedParameter(UninitializedTensorMixin, Parameter):

    def __init__(self, requires_grad: bool = True) -> None:
        dummy_tensor = nova.empty(0)
        Parameter.__init__(self, data=dummy_tensor, requires_grad=requires_grad)

    def materialize(self, size: Size, dtype: Optional[Dtype] = None) -> Parameter:
        tensor = nova.empty(size, dtype=dtype)
        return Parameter(tensor)

    def __repr__(self) -> str:
        return "<UninitializedParameter>"


@registry_class
class UninitializedBuffer(UninitializedTensorMixin, Buffer):

    def __init__(self, persistent: bool = True) -> None:
        dummy_tensor = nova.empty(0)
        Buffer.__init__(self, data=dummy_tensor, persistent=persistent)

    def materialize(self, size: Size, dtype: Optional[Dtype] = None) -> Buffer:
        tensor = nova.empty(size, dtype=dtype)
        return Buffer(tensor, persistent=self.persistent)

    def __repr__(self) -> str:
        return "<UninitializedBuffer>"


def is_lazy(param: Any) -> bool:
    """
    Returns whether ``param`` is an ``UninitializedParameter`` or ``UninitializedBuffer``.

    Args:
        param (Any): the input to check.
    """
    return isinstance(param, UninitializedTensorMixin)
