from __future__ import annotations
import nova
import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING, Any, Callable, Optional
from nova._interfaces._base_tensor import TensorBase
from nova.utils import registry_class

if TYPE_CHECKING:
    from nova.autograd.function import Function
    from nova._typing import Dtype
    from nova.autograd.engine import Context


@registry_class
class Tensor(TensorBase):
    __slots__ = [
        "_data_internal",
        "_dtype_internal",
        "requires_grad",
        "grad_fn",
        "grad",
        "copy",
        "_is_leaf",
        "_backward_hooks",
        "_retain_grad",
        "_inputs",
        "_ctx",
    ]

    _data_internal: ndarray
    _dtype_internal: Dtype
    requires_grad: bool
    grad_fn: Optional["Function"]
    grad: Optional[ndarray]
    copy: bool
    _is_leaf: bool
    _backward_hooks: list[Callable[[ndarray], Optional[ndarray]]]
    _retain_grad: bool
    _inputs: list[Tensor | Any]
    _ctx: "Context"

    def __init__(
        self,
        data: Any,
        dtype: Optional[Dtype] = None,
        requires_grad: bool = False,
        grad_fn: Optional[Function] = None,
        copy: bool = True,
    ):
        self._dtype_internal: Dtype = dtype if dtype is not None else nova.float32

        if isinstance(data, Tensor):
            if copy:
                data = data.data.astype(self._dtype_internal, copy=True)
            else:

                if data.data.dtype != self._dtype_internal:

                    data = data.data.astype(self._dtype_internal)
                else:
                    data = data.data

        elif isinstance(data, ndarray):
            if copy:
                data = data.astype(self._dtype_internal, copy=True)
            else:
                if data.dtype != self._dtype_internal:
                    data = data.astype(self._dtype_internal)
                else:
                    data = data
        else:
            data = np.array(data, dtype=self._dtype_internal)

        self.data: ndarray = data
        self.requires_grad: bool = requires_grad
        self.grad_fn: Optional[Function] = (
            grad_fn if requires_grad and nova.is_grad_enabled() else None
        )
        self.grad: Optional[ndarray] = None
        self._is_leaf: bool = True if grad_fn is None else False
        self._retain_grad: bool = False
        self._backward_hooks: list[Callable[[ndarray], Optional[ndarray]]] = []

    def __repr__(self) -> str:
        prefix = "tensor("
        tensor_str = np.array2string(self.data, separator=", ", prefix=prefix)

        result = f"{prefix}{tensor_str}"
        result += f", requires_grad={self.requires_grad}"
        result += f", grad_fn={repr(self.grad_fn)}"

        dtype_str = np.dtype(self.dtype).name
        result += f", dtype={dtype_str}"
        result += ")"
        return result

    def __str__(self) -> str:
        prefix = "tensor("

        array_str = np.array2string(self.data, separator=", ", prefix=prefix)

        result = f"{prefix}{array_str}"
        if self.requires_grad:
            result += f", requires_grad={self.requires_grad}"
        result += ")"
        return result
