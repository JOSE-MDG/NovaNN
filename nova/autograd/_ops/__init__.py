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

from ._basic import (
    Add,
    Sub,
    Mul,
    Div,
    DivInt,
    Mod,
    Floor,
    Pow,
    Exp,
    Log,
    Sqrt,
    Neg,
    Sign,
    Abs,
    Ceil,
)

from ._activation import *  # noqa: F403
from ._linalg import *  # noqa: F403
from ._loss import *  # noqa: F403
from ._indexing import *  # noqa: F403
from .views import *  # noqa: F403
from ._reduction import *  # noqa: F403
from ._manipulation import *  # noqa: F403
from ._normalization import *  # noqa: F403
from ._trigonometric import *  # noqa: F403
from ._comparison import *  # noqa: F403
from ._creation import *  # noqa: F403

__all__ = [
    "Add",
    "Sub",
    "Mul",
    "Div",
    "DivInt",
    "Mod",
    "Floor",
    "Pow",
    "Exp",
    "Log",
    "Sqrt",
    "Neg",
    "Sign",
    "Abs",
    "Ceil",
]

__all__ += _activation.__all__  # noqa: F405
__all__ += _linalg.__all__  # noqa: F405
__all__ += _loss.__all__  # noqa: F405
__all__ += _indexing.__all__  # noqa: F405
__all__ += views.__all__  # noqa: F405
__all__ += _reduction.__all__  # noqa: F405
__all__ += _manipulation.__all__  # noqa: F405
__all__ += _normalization.__all__  # noqa: F405
__all__ += _trigonometric.__all__  # noqa: F405
__all__ += _comparison.__all__  # noqa: F405
__all__ += _loss.__all__  # noqa: F405
