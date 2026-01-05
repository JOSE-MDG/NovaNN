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

from ._activation import *
from ._linalg import *
from ._indexing import *
from .views import *
from ._reduction import *
from ._manipulation import *
from ._trigonometric import *
from ._comparison import *
from ._creation import *

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

__all__ += _activation.__all__
__all__ += _linalg.__all__
__all__ += _indexing.__all__
__all__ += views.__all__
__all__ += _reduction.__all__
__all__ += _manipulation.__all__
__all__ += _trigonometric.__all__
__all__ += _comparison.__all__
