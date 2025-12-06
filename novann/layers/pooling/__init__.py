"""Max Pooling and Global Average Pooling layers."""

from .gap import GlobalAvgPool1d, GlobalAvgPool2d
from .maxpool import MaxPool1d, MaxPool1d

__all__ = ["MaxPool1d", "MaxPool2d", "GlobalAvgPool1d", "GlobalAvgPool2d"]
