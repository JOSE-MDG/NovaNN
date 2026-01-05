"""
Autograd execution engine.

This module contains the internal components responsible for
executing the backward pass of the computation graph.

It defines:
- `Context`: storage for forward-pass data needed during backpropagation
- `_backward`: the core reverse-mode automatic differentiation engine

This package is intended for internal use by the autograd system.
"""

from .context import Context
from .engine import _backward

__all__ = ["Context", "_backward"]
