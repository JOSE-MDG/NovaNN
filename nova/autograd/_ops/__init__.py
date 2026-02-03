"""
Autograd operations registry.

This module aggregates and exposes all differentiable operations
supported by the autograd engine. Each operation is implemented
as a subclass of `Function` and registered through the internal
operator registry.

The operations are organized by category:
- Basic arithmetic
- Activations
- Linear algebra
- Indexing and views
- Reductions
- Tensor manipulation
- Trigonometric functions
- Comparison operators
- Tensor creation

Only the symbols listed in `__all__` are considered part of the
public autograd API.
"""

from . import (
    _activation,
    _arithmetic,
    _comparison,
    _convolution,
    _creation,
    _indexing,
    _linalg,
    _loss,
    _manipulation,
    _normalization,
    _random,
    _reduction,
    _trigonometric,
    _view,
)

from ._activation import *  # noqa: F403
from ._arithmetic import *  # noqa: F403
from ._comparison import *  # noqa: F403
from ._convolution import *  # noqa: F403
from ._creation import *  # noqa: F403
from ._indexing import *  # noqa: F403
from ._linalg import *  # noqa: F403
from ._loss import *  # noqa: F403
from ._manipulation import *  # noqa: F403
from ._normalization import *  # noqa: F403
from ._random import *  # noqa: F403
from ._reduction import *  # noqa: F403
from ._trigonometric import *  # noqa: F403
from ._view import *  # noqa: F403


modules = [
    _activation,
    _arithmetic,
    _comparison,
    _convolution,
    _creation,
    _indexing,
    _linalg,
    _loss,
    _manipulation,
    _normalization,
    _random,
    _reduction,
    _trigonometric,
    _view,
]

__all__ = []

for m in modules:
    __all__.extend(m.__all__)

__all__ = list(dict.fromkeys(__all__))
