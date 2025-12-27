from __future__ import annotations
import builtins
from typing import Any, TYPE_CHECKING, Optional
from .dtypes import *
from .utils import registry_class, ensure_tensor, registry_op
from ._internal._binding import bootstrap_to
from ._tensor import Tensor
from .autograd.grad_mode import is_grad_enabled
from .autograd._ops._creation import *

if TYPE_CHECKING:
    from nova._typing import Dtype
    from nova.autograd.function import Function

__all__ = [
    # Dtypes
    "uint8",
    "int8",
    "short",
    "int",
    "long",
    "half",
    "float32",
    "double",
    "float128",
    "bool",
    # Tensor
    "Tensor",
    # grad_state
    "is_grad_enabled",
    # registers
    "registry_class",
    "registry_op",
    # ensure_tensor
    "ensure_tensor",
    # bootstrap
    "bootstrap_to",
]

__all__.extend(__all__)

__version__ = "3.0.0"


def tensor(
    data: Any,
    dtype: Optional[Dtype] = None,
    requires_grad: builtins.bool = False,
    grad_fn: Optional[Function] = None,
    copy: builtins.bool = True,
):
    return Tensor(
        data, dtype=dtype, requires_grad=requires_grad, grad_fn=grad_fn, copy=copy
    )


bootstrap_to(Tensor)
