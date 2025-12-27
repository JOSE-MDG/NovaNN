from __future__ import annotations
import nova
from typing import Any
from nova import Tensor


class Parameter(Tensor):

    __slots__ = ["is_bn_param"]

    def __init__(self, data: Any, requires_grad: bool = True) -> None:
        super().__init__(data=data, requires_grad=requires_grad, dtype=nova.float32)

        self.is_bn_param: bool = False


class Buffer(Tensor):

    __slots__ = ["persistent"]

    def __init__(self, data: Any, *, persistent: bool = True):
        super().__init__(data=data)
        self.persistent = persistent
