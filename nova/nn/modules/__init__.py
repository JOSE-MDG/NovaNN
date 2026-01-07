"""
Neural network modules for building deep learning models.

This package provides the building blocks for constructing neural networks in Nova,
similar to PyTorch's torch.nn module. It includes:

- **Core abstractions**: Module base class for all neural network components
- **Layers**: Linear, convolutional, normalization, and pooling layers
- **Activations**: Non-linear activation functions (ReLU, Sigmoid, etc.)
- **Loss functions**: Criterion modules for training (MSE, CrossEntropy, etc.)
- **Containers**: Sequential and other containers for composing modules
- **Utilities**: Dropout, Flatten, and other utility layers

All modules inherit from the Module base class and implement the forward() method,
enabling automatic differentiation and parameter management.

Examples:
    >>> import nova.nn as nn

    >>> # Build a simple feedforward network
    >>> model = nn.Sequential(
    ...     nn.Linear(784, 128),
    ...     nn.ReLU(),
    ...     nn.Dropout(0.5),
    ...     nn.Linear(128, 10)
    ... )

    >>> # Build a convolutional network
    >>> class ConvNet(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
    ...         self.pool = nn.MaxPool2d(2, 2)
    ...         self.fc = nn.Linear(32 * 14 * 14, 10)
    ...
    ...     def forward(self, x):
    ...         x = self.pool(nn.functional.relu(self.conv1(x)))
    ...         x = nn.Flatten()(x)
    ...         x = self.fc(x)
    ...         return x

    >>> # Define loss and train
    >>> criterion = nn.CrossEntropyLoss()
    >>> optimizer = nova.optim.Adam(model.parameters())
    >>>
    >>> for data, target in dataloader:
    ...     output = model(data)
    ...     loss = criterion(output, target)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()

Note:
    Most layers have both standard and "Lazy" variants. Lazy modules automatically
    infer input dimensions on first forward pass, reducing boilerplate code.
"""

from .module import Module
from .linear import Linear, LazyLinear
from .container import Sequential
from .activation import (
    ReLU,
    LeakyReLU,
    PReLU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,
)
from .lazy import LazyModuleMixin
from .flatten import Flatten
from .batchnorm import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    LazyBatchNorm1d,
    LazyBatchNorm2d,
    LazyBatchNorm3d,
)
from .conv import Conv1d, Conv2d, Conv3d, LazyConv1d, LazyConv2d, LazyConv3d
from .dropout import Dropout, Dropout2d, Dropout3d
from .layernorm import LayerNorm
from .pooling import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    GlobalAvgPool1d,
    GlobalAvgPool2d,
    GlobalAvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)
from .loss import (
    MSELoss,
    L1Loss,
    SmoothL1Loss,
    BCELoss,
    BCEWithLogitsLoss,
    NLLLoss,
    CrossEntropyLoss,
    KLDivLoss,
)

__all__ = [
    # Core
    "Module",
    "LazyModuleMixin",
    # Containers
    "Sequential",
    # Linear layers
    "Linear",
    "LazyLinear",
    # Convolutional layers
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    # Normalization layers
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LayerNorm",
    # Pooling layers
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "GlobalAvgPool1d",
    "GlobalAvgPool2d",
    "GlobalAvgPool3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    # Activation functions
    "ReLU",
    "LeakyReLU",
    "PReLU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LogSoftmax",
    # Regularization
    "Dropout",
    "Dropout2d",
    "Dropout3d",
    # Utility layers
    "Flatten",
    # Loss functions
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "NLLLoss",
    "CrossEntropyLoss",
    "KLDivLoss",
]
