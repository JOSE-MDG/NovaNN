"""Sequential container for chaining neural network layers.

This module provides the Sequential class, a container that allows stacking
multiple layers to form a complete neural network model with a linear forward
pass through each layer in sequence.
"""

from .nn import Sequential

__all__ = ["Sequential"]
