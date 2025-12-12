"""NovaNN: A lightweight neural network framework from scratch.

NovaNN provides a modular, NumPy-based deep learning framework with automatic
differentiation. It includes layers, loss functions, optimizers, and utilities
for building, training, and evaluating neural networks.
The framework is designed for educational purposes and practical experimentation,
offering transparent implementations of core deep learning concepts.
"""

from . import optim
from . import functional
from .losses import BinaryCrossEntropy, CrossEntropyLoss, MAE, MSE
from .model import Sequential
from .core import init
from .metrics import metrics

from .layers import (
    LeakyReLU,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    Linear,
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    MaxPool2d,
    MaxPool1d,
    Flatten,
    GlobalAvgPool1d,
    GlobalAvgPool2d,
    Dropout,
)

__all__ = [
    "LeakyReLU",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "Linear",
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv1d",
    "Conv2d",
    "MaxPool2d",
    "MaxPool1d",
    "Flatten",
    "GlobalAvgPool1d",
    "GlobalAvgPool2d",
    "Dropout",
    "Sequential",
    "functional",
    "optim",
    "init",
    "metrics",
    "BinaryCrossEntropy",
    "CrossEntropyLoss",
    "MAE",
    "MSE",
]
__version__ = "3.0.0"
