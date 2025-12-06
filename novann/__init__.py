"""NovaNN: A lightweight neural network framework from scratch.

NovaNN provides a modular, NumPy-based deep learning framework with automatic
differentiation. It includes layers, loss functions, optimizers, and utilities
for building, training, and evaluating neural networks.

The framework is designed for educational purposes and practical experimentation,
offering transparent implementations of core deep learning concepts.
"""

from . import *

__all__ = [
    "core",
    "layers",
    "losses",
    "model",
    "module",
    "optim",
    "utils",
]

__version__ = "1.1.0"
