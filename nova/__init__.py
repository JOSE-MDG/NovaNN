from __future__ import annotations
import builtins
from typing import Any, TYPE_CHECKING, Optional
from .dtypes import *
from .utils import registry_class, ensure_tensor, registry_op
from ._internal._binding import bootstrap_to
from ._tensor import Tensor
import nova.autograd as autograd
from .autograd.grad_mode import is_grad_enabled, enable_grad, no_grad
from .autograd._ops._creation import *
from .autograd._ops._creation import __all__ as creation_all
from .serialization import save, load

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
    # autograd
    "autograd",
    # grad_state
    "is_grad_enabled",
    # registers
    "registry_class",
    "registry_op",
    # ensure_tensor
    "ensure_tensor",
    # bootstrap
    "bootstrap_to",
    # grad handling
    "no_grad",
    "enable_grad",
    # serialization
    "save",
    "load",
]

__all__.extend(creation_all)

__version__ = "3.0.0"


def tensor(
    data: Any,
    dtype: Optional[Dtype] = None,
    requires_grad: builtins.bool = False,
    grad_fn: Optional[Function] = None,
) -> Tensor:
    """
    Creates a tensor from data.

    This is the primary factory function for creating tensors, similar to
    torch.tensor(). It wraps the Tensor constructor with a more convenient
    interface commonly used in deep learning code.

    Args:
        data: Input data. Can be a list, tuple, numpy array, scalar, or
            another Tensor.
        dtype: Desired data type for the tensor. If None, infers from data
            or defaults to float32.
        requires_grad: If True, tracks operations on this tensor for automatic
            differentiation. Default is False.
        grad_fn: Function that created this tensor (internal use only, typically
            None for user-created tensors).

    Returns:
        A new Tensor containing the provided data.

    Examples:
        >>> # Create from list
        >>> x = nova.tensor([1, 2, 3])
        >>> print(x)
        tensor([1., 2., 3.])

        >>> # Create with specific dtype
        >>> x = nova.tensor([1, 2, 3], dtype=nova.long)
        >>> print(x.type()) # or x.dtype
        'nova.long'

        >>> # Create with gradient tracking
        >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
        >>> y = (x ** 2).sum()
        >>> y.backward()
        >>> print(x.grad)
        [2., 4.]

        >>> # Create from numpy array
        >>> import numpy as np
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> x = nova.tensor(arr, dtype=nova.float32)

    Notes:
        - For creating tensors with specific patterns (zeros, ones, randn, etc.),
          use the dedicated factory functions: nova.zeros(), nova.ones(),
          nova.randn(), etc.
        - This function always creates a new tensor (copies data by default).
        - Scalar inputs are automatically converted to 0-dimensional tensors.
    """
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, grad_fn=grad_fn)


# Bootstrap all operations from YAML to Tensor class
# This dynamically binds methods like __add__, relu, matmul, etc.
bootstrap_to(Tensor)
