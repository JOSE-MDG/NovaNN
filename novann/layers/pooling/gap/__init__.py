"""Global average pooling layers for 1D and 2D data."""

from .global_avg_pool1d import GlobalAvgPool1d
from .global_avg_pool2d import GlobalAvgPool2d

__all__ = ["GlobalAvgPool1d", "GlobalAvgPool2d"]
