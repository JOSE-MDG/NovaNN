from __future__ import annotations
import nova
from typing import Any
from nova import Tensor
from nova.utils import registry_class


@registry_class
class Parameter(Tensor):

    __slots__ = ["is_bn_param"]

    def __init__(self, data: Any, requires_grad: bool = True) -> None:
        super().__init__(
            data=data, requires_grad=requires_grad, dtype=nova.float32, copy=False
        )

        self.is_bn_param: bool = False


@registry_class
class Buffer(Tensor):

    __slots__ = ["persistent"]

    def __init__(self, data: Any, *, persistent: bool = True):
        super().__init__(data=data, requires_grad=False)
        self.persistent = persistent
