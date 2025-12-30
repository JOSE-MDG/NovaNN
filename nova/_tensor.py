from __future__ import annotations
import nova
import numpy as np
import traceback
from numpy import ndarray
from typing import TYPE_CHECKING, Any, Optional, Self
from nova._interfaces._base_tensor import TensorBase
from nova.utils import registry_class, ensure_tensor
from nova.utils.log_config import logger
from nova.autograd.engine import _backward

if TYPE_CHECKING:
    from nova.autograd.function import Function
    from nova._typing import Dtype, Hook, Dim
    from nova.autograd.engine import Context
    from nova.autograd.utils.hooks import HooksHandle


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
        "rank",
    ]

    _data_internal: ndarray
    _dtype_internal: Dtype
    requires_grad: bool
    grad_fn: Optional[Function]
    grad: Optional[ndarray]
    copy: bool
    _is_leaf: bool
    _backward_hooks: list[Hook]
    _retain_grad: bool
    _inputs: list[Tensor | Any]
    rank: int
    _ctx: Context

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
        super().__init__()
        self.data: ndarray = data
        self.requires_grad: bool = requires_grad
        self.grad_fn: Optional[Function] = grad_fn
        self.rank: int = 0
        self._inputs: list[Tensor] = []
        self._ctx: Optional[Context] = None
        self.grad: Optional[ndarray] = None
        self._is_leaf: bool = True if grad_fn is None else False
        self._retain_grad: bool = False
        self._backward_hooks: list[Hook] = []

    def __eq__(self, other) -> Tensor:
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data == other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __req__(self, other) -> Tensor:
        return self.__eq__(other)

    def __ne__(self, other) -> Tensor:
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data != other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __rne__(self, other) -> Tensor:
        return self.__ne__(other)

    def __lt__(self, other) -> Tensor:
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data < other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __rlt__(self, other) -> Tensor:
        return self.__gt__(other)

    def __le__(self, other) -> Tensor:
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data <= other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __rle__(self, other) -> Tensor:
        return self.__ge__(other)

    def __gt__(self, other) -> Tensor:
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data > other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __rgt__(self, other) -> Tensor:
        return self.__lt__(other)

    def __ge__(self, other) -> Tensor:
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data >= other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __rge__(self, other) -> Tensor:
        return self.__le__(other)

    def __invert__(self) -> Tensor:
        return Tensor(~self.data, dtype=nova.bool, requires_grad=False)

    def __hash__(self):
        return id(self)

    def __len__(self) -> int:
        return len(self.data)

    def argmax(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return Tensor(
            self.data.argmax(axis=dim, keepdims=keepdims),
            dtype=self.dtype,
            requires_grad=False,
        )

    def argmin(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return Tensor(
            self.data.argmin(axis=dim, keepdims=keepdims),
            dtype=self.dtype,
            requires_grad=False,
        )

    def argsort(self, dim: Optional[Dim] = None, kind=None, order=None) -> Tensor:
        return Tensor(
            self.data.argsort(axis=dim, kind=kind, order=order),
            dtype=self.dtype,
            requires_grad=False,
        )

    def argwhere(self) -> Tensor:
        return Tensor(np.argwhere(self.data), dtype=self.dtype, requires_grad=False)

    def var(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        from nova.autograd._ops import var

        return var(self, dim=dim, keepdims=keepdims)

    def std(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        from nova.autograd._ops import std

        return std(self, dim=dim, keepdims=keepdims)

    def detach(self) -> Tensor:
        return Tensor(self.data, dtype=self.dtype, requires_grad=False)

    def item(self) -> float | int:
        if self.numel() > 1:
            raise ValueError(
                "only one element tensors can be converted to Python scalars."
            )

        return self.data.item()

    def all(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return Tensor(
            self.data.all(dim, keepdims=keepdims), dtype=nova.bool, requires_grad=False
        )

    def any(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        return Tensor(
            self.data.any(dim, keepdims=keepdims), dtype=nova.bool, requires_grad=False
        )

    # --- Operaciones de Casting (Out-of-place) ---

    def to(self, dtype: Dtype) -> Tensor:
        return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad)

    def float(self) -> Tensor:
        return self.to(nova.float32)

    def double(self) -> Tensor:
        return self.to(nova.double)

    def int(self) -> Tensor:
        return self.to(nova.int)

    def long(self) -> Tensor:
        return self.to(nova.long)

    def numpy(self) -> ndarray:
        return self.data.copy()

    def allclose(self, other, rtol=1e-5, atol=1e-8, equal_nan: bool = False) -> bool:
        return np.allclose(
            self.data,
            ensure_tensor(other).data,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    # inplace-methods

    def normal_(self, mean: float = 0, std: float = 1) -> None:
        if self.requires_grad:
            raise RuntimeError(
                f"Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data = np.random.normal(loc=mean, scale=std, size=self.data.shape).astype(
            self.dtype
        )

    def uniform_(self, low: float = -1, high: float = 1) -> None:
        if self.requires_grad:
            raise RuntimeError(
                f"Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data = np.random.uniform(low=low, high=high, size=self.data.shape).astype(
            self.dtype
        )

    def random_(self) -> None:
        if self.requires_grad:
            raise RuntimeError(
                f"Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data = np.random.rand(self.data.shape).astype(self.dtype)

    def zero_(self) -> None:
        if self.requires_grad:
            raise RuntimeError(
                f"Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data.fill(0.0)

    def ones_(self) -> None:
        if self.requires_grad:
            raise RuntimeError(
                f"Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data.fill(1.0)

    def fill_(self, value: Any) -> None:
        if self.requires_grad:
            raise RuntimeError(
                f"Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data.fill(value)

    def copy_(self, src: Tensor) -> Self[Tensor]:
        src = ensure_tensor(src)

        src_data = src.data

        if src.data.shape != self.data.shape:
            raise ValueError(f"Shape mismatch: {self.data.shape} vs {src_data.shape}")

        self.data[:] = src_data

        return self

    def requires_grad_(self, mode: bool) -> None:
        self.requires_grad = mode

    def retain_grad(self) -> Self[Tensor]:

        if not self.requires_grad:
            raise RuntimeError("Only tensors with requires_grad can retain gradients")

        self._retain_grad = True

        return self

    def register_hook(self, hook: Hook) -> HooksHandle:

        if not self.requires_grad:
            raise RuntimeError(
                "Cannot register a hook on a tensor that doesn't require gradients"
            )

        self._backward_hooks.append(hook)
        return HooksHandle(self._backward_hooks, hook)

    def zero_grad(self, set_to_none: bool = True) -> None:
        if set_to_none:
            self.grad = None
        else:
            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=self.data.dtype)
            else:
                self.grad.fill(0.0)

    def _apply_hooks(self, grad: ndarray) -> ndarray:

        try:
            current_grad = grad

            for hook in self._backward_hooks:

                result = hook(current_grad)

                if result is not None:

                    if result.shape != grad:
                        raise ValueError(
                            f"Hook returned gradient with shape {result.shape}, "
                            f"expected {current_grad.shape}"
                        )

                    current_grad = result

            return current_grad
        except Exception as e:
            lines = [line for line in traceback.format_exception(e)]
            logger.error("Error executing backward hook\n\n")
            print(*lines)

    def backward(
        self, gradient: Optional[ndarray | Tensor] = None, retain_graph: bool = False
    ) -> None:

        if not self.requires_grad:
            raise RuntimeError(
                "An attempt was made to calculate gradients for a tensor that does not require gradients."
            )

        if gradient is None:
            if self.numel() > 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs"
                )
            gradient = np.ones_like(self.data, dtype=nova.float32)
        else:
            gradient = (
                gradient.data.astype(nova.float32)
                if isinstance(gradient, Tensor)
                else np.asarray(gradient, dtype=nova.float32)
            )
        try:
            _backward(self, gradient=gradient, retain_graph=retain_graph)
        except Exception as e:
            lines = [line for line in traceback.format_exception(e)]
            logger.error("Error during the graph creation\n\n")
            print(*lines)

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
