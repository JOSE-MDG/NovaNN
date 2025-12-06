"""Base classes for neural network modules and layers.

This module defines the fundamental building blocks of the NovaNN framework:
Module (the base class for all network components), Layer (the abstract base
class for all neural network layers), and Parameters (a wrapper for trainable
parameters with gradient tracking).
"""

from .module import Module, Parameters
from .layer import Layer

__all__ = ["Module", "Parameters", "Layer"]
