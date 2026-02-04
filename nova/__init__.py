"""
NovaNN Core API
===============

This module defines the **public top-level API of NovaNN**.

It acts as the main entry point of the framework, exposing the core abstractions
and utilities required for building, training, and analyzing neural networks.
The goal of this module is to provide a **clean, minimal, and explicit user-facing
interface**, while hiding the internal complexity of the framework.

At a high level, this module is responsible for:

- Exposing the `Tensor` type, which represents both data and differentiable values.
- Providing factory functions for tensor creation (e.g. `tensor`, `zeros`, `randn`, etc.).
- Making data types (`dtypes`) available at the top level.
- Managing global autograd state (enabling/disabling gradient tracking).
- Registering and binding operations dynamically to the `Tensor` class.
- Exposing serialization utilities (`save`, `load`).
- Initializing the operator system via a bootstrap mechanism.

Design Philosophy
-----------------
NovaNN follows a **PyTorch-inspired but minimalistic design**, prioritizing:

- Explicitness over magic
- Computational graphs
- Predictable autograd behavior
- Low overhead for CPU-based experimentation
- Readable and inspectable internals

This module intentionally re-exports selected symbols from internal submodules
to provide a convenient and discoverable API, while keeping the internal layout
modular and well-isolated.

Public API Surface
------------------
The symbols exposed here represent the **stable API** intended for users.
Internal modules (prefixed with `_`) are not considered part of the public API
and may change without notice.

Key components exposed:

- `Tensor`: Core tensor abstraction with autograd support
- `tensor(...)`: Primary tensor factory function
- Dtypes: `float32`, `int`, `bool`, etc.
- Autograd utilities: `is_grad_enabled`, `enable_grad`, `no_grad`
- Serialization: `save`, `load`
- Operation registries and dynamic bindings

Dynamic Operation Binding
-------------------------
At import time, NovaNN dynamically binds operations defined in YAML and Python
implementations to the `Tensor` class. This allows operators such as arithmetic,
activation functions, indexing, and linear algebra routines to be attached
without hardcoding them into the class definition.

This design enables:
- Clear separation between tensor structure and operations
- Easier extensibility
- Fine-grained control over operator registration

This module is intentionally lightweight in logic and heavy in orchestration,
serving as the connective layer between NovaNN internal systems and its users.
"""

from __future__ import annotations
import builtins
from typing import Any, TYPE_CHECKING, Optional
from .dtypes import *  # noqa: F403
from .dtypes import __all__ as dtypes_all
from .exceptions import *  # noqa: F403, F401
from . import core
from .utils import registry_class, registry_op, ensure_tensor
from ._internal._binding import bootstrap_to
from ._tensor import Tensor
from . import autograd
from .autograd.grad_mode import is_grad_enabled, enable_grad, no_grad
from .autograd._ops._creation import *  # noqa: F403
from .autograd._ops._creation import __all__ as creation_all
from .autograd._ops._random import *  # noqa: F403
from .autograd._ops._random import __all__ as random_all
from . import nn
from . import optim
from . import metrics
from . import utils
from .serialization import save, load

if TYPE_CHECKING:
    from nova._typing import Dtype
    from nova.autograd.function import Function

__all__ = [
    # ---- Core Modules ----
    "autograd",
    "core",
    "nn",
    "optim",
    "metrics",
    "utils",
    # ---- Tensor Class ----
    "Tensor",
    # ---- Autograd State Management ----
    "is_grad_enabled",
    "enable_grad",
    "no_grad",
    # ---- Registry Functions ----
    "registry_class",
    "registry_op",
    # ---- Utilities ----
    "ensure_tensor",
    # ---- Internal Systems ----
    "bootstrap_to",
    # ---- Serialization ----
    "save",
    "load",
]

# Extend with dtype names
__all__.extend(dtypes_all)

# Extend with creation function names
__all__.extend(creation_all)

# Extend with random function names
__all__.extend(random_all)

# VERSION
__version__ = "4.0.1"


# TENSOR FACTORY FUNCTION
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
# MUST be called AFTER all imports are complete
bootstrap_to(Tensor)
