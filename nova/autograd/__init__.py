"""
Automatic differentiation module.

This package provides the core automatic differentiation
infrastructure of Nova, including:

- Differentiable operations (`autograd._ops`)
- Gradient computation via reverse-mode autodiff
- The `grad` interface for user-facing gradient evaluation

This module exposes the public autograd API.
"""

from .grad import grad
from nova.autograd import _ops

__all__ = ["grad", "_ops"]
