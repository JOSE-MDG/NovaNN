"""
Comparison and logical operations for Nova.

This module provides element-wise minimum/maximum and conditional
selection (where) with broadcasting-aware autograd support.
"""

from .comparison import Maximum, Minimum, Where

__all__ = ["Maximum", "Minimum", "Where"]
