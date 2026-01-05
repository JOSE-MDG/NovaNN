from __future__ import annotations
import nova.nn.functional as F
from typing import TYPE_CHECKING
from nova.nn.modules import Module

if TYPE_CHECKING:
    from nova import Tensor


class Flatten(Module):
    """Flattens a contiguous range of dimensions into a single dimension.

    This layer is commonly used to transition from convolutional layers to fully
    connected layers in neural networks. It reshapes the input by flattening
    dimensions from ``start_dim`` to ``end_dim`` (inclusive) into a single dimension.

    Args:
        start_dim: First dimension to flatten. Default: 1 (preserves batch dimension)
        end_dim: Last dimension to flatten. Default: -1 (flattens to the end)

    Shape:
        - Input: :math:`(*, S_{\\text{start}}, ..., S_{\\text{end}}, *)`
        - Output: :math:`(*, \\prod_{i=\\text{start}}^{\\text{end}} S_i, *)`

        where :math:`*` means any number of dimensions outside the flattened range

    Examples::

        >>> # Typical CNN to FC transition (flatten all except batch)
        >>> flatten = Flatten()
        >>> x = nova.randn(32, 3, 28, 28)  # (batch, channels, height, width)
        >>> output = flatten(x)
        >>> print(output.shape)  # (32, 2352)  where 2352 = 3*28*28

        >>> # Complete CNN example
        >>> class ConvNet(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = Conv2d(1, 32, 3)
        ...         self.conv2 = Conv2d(32, 64, 3)
        ...         self.flatten = Flatten()
        ...         self.fc1 = Linear(64 * 24 * 24, 128)
        ...         self.fc2 = Linear(128, 10)
        ...
        ...     def forward(self, x):
        ...         x = relu(self.conv1(x))
        ...         x = relu(self.conv2(x))
        ...         x = self.flatten(x)  # (N, 64, 24, 24) -> (N, 36864)
        ...         x = relu(self.fc1(x))
        ...         x = self.fc2(x)
        ...         return x

        >>> # Custom flattening range
        >>> flatten = Flatten(start_dim=2, end_dim=3)
        >>> x = nova.randn(10, 5, 3, 4, 7)
        >>> output = flatten(x)
        >>> print(output.shape)  # (10, 5, 12, 7)  where 12 = 3*4

        >>> # Flatten everything (including batch)
        >>> flatten = Flatten(start_dim=0)
        >>> x = nova.randn(2, 3, 4, 5)
        >>> output = flatten(x)
        >>> print(output.shape)  # (120,)  where 120 = 2*3*4*5

        >>> # Equivalent to view() but more explicit
        >>> x = nova.randn(8, 16, 7, 7)
        >>> manual = x.view(8, -1)
        >>> auto = Flatten()(x)
        >>> assert manual.shape == auto.shape  # Both (8, 784)

    Note:
        By default (start_dim=1), the batch dimension is preserved, which is
        the most common use case when transitioning from convolutional to
        fully connected layers.
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim: int = start_dim
        self.end_dim: int = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """Flattens the input tensor along specified dimensions.

        Args:
            input: Input tensor to flatten.

        Returns:
            Flattened tensor with dimensions from start_dim to end_dim merged.
        """
        return F.flatten(input, start_dim=self.start_dim, end_dim=self.end_dim)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"
