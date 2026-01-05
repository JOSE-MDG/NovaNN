from __future__ import annotations
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.utils import _single, _pair, _triple
from nova.nn.modules import Module

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import KernelSize, Padding, Stride, Dilation


class GlobalAvgPool1d(Module):
    """Applies global average pooling over the temporal dimension.

    Reduces each channel to a single value by averaging across the entire
    sequence length. Commonly used as the final pooling layer before
    classification in temporal CNNs.

    .. math::
        \\text{output}_c = \\frac{1}{L} \\sum_{i=0}^{L-1} \\text{input}_{c,i}

    Shape:
        - Input: :math:`(N, C, L)` where N is batch size, C is channels, L is length
        - Output: :math:`(N, C, 1)` - each channel reduced to single value

    Examples::

        >>> # Global pooling for 1D sequences
        >>> global_pool = GlobalAvgPool1d()
        >>> x = nova.randn(8, 64, 100)  # (batch, channels, length)
        >>> output = global_pool(x)
        >>> print(output.shape)  # (8, 64, 1)

        >>> # Common in temporal CNNs for classification
        >>> class TemporalCNN(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = Conv1d(1, 32, kernel_size=3)
        ...         self.conv2 = Conv1d(32, 64, kernel_size=3)
        ...         self.global_pool = GlobalAvgPool1d()
        ...         self.fc = Linear(64, 10)
        ...
        ...     def forward(self, x):
        ...         x = relu(self.conv1(x))
        ...         x = relu(self.conv2(x))
        ...         x = self.global_pool(x)  # (N, 64, L) -> (N, 64, 1)
        ...         x = x.squeeze(-1)  # (N, 64)
        ...         x = self.fc(x)
        ...         return x

    Note:
        Equivalent to AvgPool1d with kernel_size equal to the input length,
        but more efficient and handles variable-length inputs.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies global average pooling.

        Args:
            input: Input tensor of shape (N, C, L).

        Returns:
            Pooled tensor of shape (N, C, 1).
        """
        return F.global_avg_pool1d(input)


class GlobalAvgPool2d(Module):
    """Applies global average pooling over the spatial dimensions.

    Reduces each channel to a single value by averaging across the entire
    spatial extent (height and width). This is the standard approach for
    CNNs like ResNet and MobileNet to replace fully connected layers.

    .. math::
        \\text{output}_c = \\frac{1}{H \\times W} \\sum_{i=0}^{H-1} \\sum_{j=0}^{W-1} \\text{input}_{c,i,j}

    Shape:
        - Input: :math:`(N, C, H, W)` where N is batch, C is channels, H is height, W is width
        - Output: :math:`(N, C, 1, 1)` - each channel reduced to single value

    Examples::

        >>> # Standard global pooling for image classification
        >>> global_pool = GlobalAvgPool2d()
        >>> x = nova.randn(32, 512, 7, 7)  # After conv layers
        >>> output = global_pool(x)
        >>> print(output.shape)  # (32, 512, 1, 1)

        >>> # ResNet-style architecture
        >>> class ResNetClassifier(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.backbone = ResNet50()
        ...         self.global_pool = GlobalAvgPool2d()
        ...         self.fc = Linear(2048, 1000)
        ...
        ...     def forward(self, x):
        ...         x = self.backbone(x)  # (N, 2048, 7, 7)
        ...         x = self.global_pool(x)  # (N, 2048, 1, 1)
        ...         x = x.view(x.size(0), -1)  # (N, 2048)
        ...         x = self.fc(x)
        ...         return x

        >>> # Compare to adaptive average pooling
        >>> x = nova.randn(16, 256, 14, 14)
        >>> gap = GlobalAvgPool2d()
        >>> out = gap(x)
        >>> print(out.shape)  # (16, 256, 1, 1)
        >>> # Equivalent to: x.mean(dim=(2, 3), keepdims=True)

    Note:
        Preferred over fully connected layers for classification as it:
        - Reduces parameters (no weights to learn)
        - Makes network resolution-agnostic
        - Acts as structural regularizer
        - Enables class activation mapping (CAM)

    Reference:
        Lin et al., "Network In Network" (2013) - Introduced global average pooling
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies global average pooling.

        Args:
            input: Input tensor of shape (N, C, H, W).

        Returns:
            Pooled tensor of shape (N, C, 1, 1).
        """
        return F.global_avg_pool2d(input)


class GlobalAvgPool3d(Module):
    """Applies global average pooling over the spatial and temporal dimensions.

    Reduces each channel to a single value by averaging across depth, height,
    and width. Used in 3D CNNs for video classification or volumetric data.

    .. math::
        \\text{output}_c = \\frac{1}{D \\times H \\times W} \\sum_{i,j,k} \\text{input}_{c,i,j,k}

    Shape:
        - Input: :math:`(N, C, D, H, W)` where D is depth, H is height, W is width
        - Output: :math:`(N, C, 1, 1, 1)` - each channel reduced to single value

    Examples::

        >>> # Global pooling for video classification
        >>> global_pool = GlobalAvgPool3d()
        >>> x = nova.randn(8, 512, 8, 7, 7)  # (batch, channels, frames, H, W)
        >>> output = global_pool(x)
        >>> print(output.shape)  # (8, 512, 1, 1, 1)

        >>> # 3D CNN for action recognition
        >>> class VideoClassifier(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv3d_1 = Conv3d(3, 64, kernel_size=3)
        ...         self.conv3d_2 = Conv3d(64, 128, kernel_size=3)
        ...         self.global_pool = GlobalAvgPool3d()
        ...         self.fc = Linear(128, 400)  # 400 action classes
        ...
        ...     def forward(self, x):
        ...         x = relu(self.conv3d_1(x))
        ...         x = relu(self.conv3d_2(x))
        ...         x = self.global_pool(x)  # Reduce spatiotemporal dims
        ...         x = x.view(x.size(0), -1)
        ...         x = self.fc(x)
        ...         return x

    Note:
        Particularly useful for 3D medical imaging (CT/MRI) and video
        understanding tasks where spatial and temporal information
        needs to be aggregated.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies global average pooling.

        Args:
            input: Input tensor of shape (N, C, D, H, W).

        Returns:
            Pooled tensor of shape (N, C, 1, 1, 1).
        """
        return F.global_avg_pool3d(input)


class AvgPool1d(Module):
    """Applies 1D average pooling over the input signal.

    Computes the average of elements in a sliding window. Useful for
    downsampling temporal sequences while preserving smooth features.

    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Zero-padding added to both sides. Default: 0

    Shape:
        - Input: :math:`(N, C, L_{in})` where L_in is input length
        - Output: :math:`(N, C, L_{out})` where
          :math:`L_{out} = \\lfloor \\frac{L_{in} + 2 \\times \\text{padding} - \\text{kernel_size}}{\\text{stride}} \\rfloor + 1`

    Examples::

        >>> # Downsample by factor of 2
        >>> pool = AvgPool1d(kernel_size=2, stride=2)
        >>> x = nova.randn(4, 16, 100)
        >>> output = pool(x)
        >>> print(output.shape)  # (4, 16, 50)

        >>> # Overlapping pooling windows
        >>> pool = AvgPool1d(kernel_size=3, stride=1, padding=1)
        >>> x = nova.randn(2, 32, 50)
        >>> output = pool(x)
        >>> print(output.shape)  # (2, 32, 50) - same size due to padding

        >>> # Typical usage in temporal CNN
        >>> x = nova.randn(8, 64, 200)
        >>> pool1 = AvgPool1d(2, 2)
        >>> x = pool1(x)  # (8, 64, 100)
        >>> pool2 = AvgPool1d(2, 2)
        >>> x = pool2(x)  # (8, 64, 50)
    """

    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Optional[Stride] = None,
        padding: Padding = 0,
    ) -> None:
        super().__init__()
        self.K = _single(kernel_size)
        self.S = _single(stride) if stride is not None else self.K
        self.P = _single(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D average pooling.

        Args:
            input: Input tensor of shape (N, C, L).

        Returns:
            Pooled tensor of shape (N, C, L_out).
        """
        return F.avg_pool1d(input, self.K, self.S, self.P)

    def extra_repr(self):
        return "kernel_size={K}, stride={S}, padding={P}".format(**self.__dict__)


class AvgPool2d(Module):
    """Applies 2D average pooling over the input image.

    Computes the average of elements in rectangular windows. Provides
    smooth downsampling compared to max pooling, preserving more
    information about feature intensity.

    Args:
        kernel_size: Size of the pooling window. Can be single int or tuple (H, W).
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Zero-padding added to both sides. Default: 0

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = \\lfloor \\frac{H_{in} + 2 \\times \\text{padding[0]} - \\text{kernel_size[0]}}{\\text{stride[0]}} \\rfloor + 1`
          :math:`W_{out} = \\lfloor \\frac{W_{in} + 2 \\times \\text{padding[1]} - \\text{kernel_size[1]}}{\\text{stride[1]}} \\rfloor + 1`

    Examples::

        >>> # Standard 2x2 pooling
        >>> pool = AvgPool2d(kernel_size=2, stride=2)
        >>> x = nova.randn(16, 64, 32, 32)
        >>> output = pool(x)
        >>> print(output.shape)  # (16, 64, 16, 16)

        >>> # Non-square kernel
        >>> pool = AvgPool2d(kernel_size=(2, 4), stride=(2, 4))
        >>> x = nova.randn(8, 32, 28, 28)
        >>> output = pool(x)
        >>> print(output.shape)  # (8, 32, 14, 7)

        >>> # With padding
        >>> pool = AvgPool2d(3, stride=1, padding=1)
        >>> x = nova.randn(4, 16, 8, 8)
        >>> output = pool(x)
        >>> print(output.shape)  # (4, 16, 8, 8) - same size

        >>> # Common in traditional CNNs (LeNet, VGG)
        >>> class SimpleCNN(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = Conv2d(3, 32, 3)
        ...         self.pool1 = AvgPool2d(2, 2)
        ...         self.conv2 = Conv2d(32, 64, 3)
        ...         self.pool2 = AvgPool2d(2, 2)
        ...
        ...     def forward(self, x):
        ...         x = relu(self.conv1(x))
        ...         x = self.pool1(x)  # Downsample
        ...         x = relu(self.conv2(x))
        ...         x = self.pool2(x)  # Downsample
        ...         return x

    Note:
        Average pooling is preferred over max pooling when:
        - You want smooth, blur-like downsampling
        - Preserving overall feature intensity is important
        - Working with heatmaps or attention maps
    """

    def __init__(
        self, kernel_size: KernelSize, stride: Stride = None, padding: Padding = 0
    ) -> None:
        super().__init__()
        self.KH, self.KW = _pair(kernel_size)
        self.SH, self.SW = _pair(stride) if stride is not None else (self.KH, self.KW)
        self.PH, self.PW = _pair(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D average pooling.

        Args:
            input: Input tensor of shape (N, C, H, W).

        Returns:
            Pooled tensor of shape (N, C, H_out, W_out).
        """
        return F.avg_pool2d(
            input, (self.KH, self.KW), (self.SH, self.SW), (self.PH, self.PW)
        )

    def extra_repr(self):
        return "kernel_size=({KH}, {KW}), stride=({SH}, {SW}), padding=({PH}, {PW})".format(
            **self.__dict__
        )


class AvgPool3d(Module):
    """Applies 3D average pooling over the input volume.

    Computes the average of elements in 3D windows. Used for downsampling
    volumetric data such as videos or 3D medical scans.

    Args:
        kernel_size: Size of the pooling window. Can be int or tuple (D, H, W).
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Zero-padding added to all sides. Default: 0

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`

    Examples::

        >>> # Downsample video by 2x in all dimensions
        >>> pool = AvgPool3d(kernel_size=2, stride=2)
        >>> x = nova.randn(4, 3, 16, 112, 112)  # (N, C, T, H, W)
        >>> output = pool(x)
        >>> print(output.shape)  # (4, 3, 8, 56, 56)

        >>> # Different pooling for temporal and spatial dims
        >>> pool = AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        >>> x = nova.randn(2, 64, 8, 32, 32)
        >>> output = pool(x)
        >>> print(output.shape)  # (2, 64, 8, 16, 16) - temporal preserved

        >>> # 3D CNN with pooling
        >>> class Video3DCNN(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = Conv3d(3, 64, 3)
        ...         self.pool1 = AvgPool3d(2, 2)
        ...         self.conv2 = Conv3d(64, 128, 3)
        ...         self.pool2 = AvgPool3d(2, 2)
        ...
        ...     def forward(self, x):
        ...         x = relu(self.conv1(x))
        ...         x = self.pool1(x)  # Spatiotemporal downsampling
        ...         x = relu(self.conv2(x))
        ...         x = self.pool2(x)
        ...         return x
    """

    def __init__(
        self, kernel_size: KernelSize, stride: Stride = None, padding: Padding = 0
    ) -> None:
        super().__init__()
        self.KD, self.KH, self.KW = _triple(kernel_size)
        self.SD, self.SH, self.SW = (
            _triple(stride) if stride is not None else (self.KD, self.KH, self.KW)
        )
        self.PD, self.PH, self.PW = _triple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D average pooling.

        Args:
            input: Input tensor of shape (N, C, D, H, W).

        Returns:
            Pooled tensor of shape (N, C, D_out, H_out, W_out).
        """
        return F.avg_pool3d(
            input,
            (self.KD, self.KH, self.KW),
            (self.SD, self.SH, self.SW),
            (self.PD, self.PH, self.PW),
        )

    def extra_repr(self):
        return "kernel_size=({KD}, {KH}, {KW}), stride=({SD}, {SH}, {SW}), padding=({PD}, {PH}, {PW})".format(
            **self.__dict__
        )


class MaxPool1d(Module):
    """Applies 1D max pooling over the input signal.

    Computes the maximum value in sliding windows. Preserves strongest
    activations and provides translation invariance for temporal features.

    Args:
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Zero-padding added to both sides. Default: 0
        dilation: Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})` where
          :math:`L_{out} = \\lfloor \\frac{L_{in} + 2 \\times \\text{padding} - \\text{dilation} \\times (\\text{kernel_size} - 1) - 1}{\\text{stride}} \\rfloor + 1`

    Examples::

        >>> # Standard max pooling
        >>> pool = MaxPool1d(kernel_size=2, stride=2)
        >>> x = nova.randn(8, 64, 100)
        >>> output = pool(x)
        >>> print(output.shape)  # (8, 64, 50)

        >>> # With dilation (dilated pooling)
        >>> pool = MaxPool1d(kernel_size=3, stride=1, dilation=2)
        >>> x = nova.randn(4, 32, 50)
        >>> output = pool(x)
        >>> print(output.shape)  # (4, 32, 46)

        >>> # Overlapping windows
        >>> pool = MaxPool1d(3, stride=1, padding=1)
        >>> x = nova.tensor([[[1, 2, 3, 4, 5]]])
        >>> output = pool(x)
        >>> print(output)  # Max in sliding windows

    Note:
        Max pooling is the most common choice for CNNs as it:
        - Provides translation invariance
        - Preserves strong activations
        - Acts as feature selector
        - Improves gradient flow compared to average pooling
    """

    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Optional[Stride] = None,
        padding: Padding = 0,
        dilation: Dilation = 1,
    ) -> None:
        super().__init__()
        self.K = _single(kernel_size)
        self.S = _single(stride) if stride is not None else self.K
        self.P = _single(padding)
        self.D = _single(dilation)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D max pooling.

        Args:
            input: Input tensor of shape (N, C, L).

        Returns:
            Pooled tensor of shape (N, C, L_out).
        """
        return F.max_pool1d(input, self.K, self.S, self.P, self.D)

    def extra_repr(self) -> str:
        return "kernel_size={K}, stride={S}, padding={P}, dilation={D}".format(
            **self.__dict__
        )


class MaxPool2d(Module):
    """Applies 2D max pooling over the input image.

    Computes the maximum value in rectangular windows. Standard downsampling
    operation in modern CNNs, preserving the strongest activations.

    Args:
        kernel_size: Size of the pooling window. Can be int or tuple (H, W).
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Zero-padding added to both sides. Default: 0
        dilation: Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`

    Examples::

        >>> # Standard 2x2 max pooling (most common)
        >>> pool = MaxPool2d(kernel_size=2, stride=2)
        >>> x = nova.randn(32, 64, 56, 56)
        >>> output = pool(x)
        >>> print(output.shape)  # (32, 64, 28, 28)

        >>> # Non-square kernel
        >>> pool = MaxPool2d(kernel_size=(3, 2), stride=(3, 2))
        >>> x = nova.randn(16, 128, 32, 32)
        >>> output = pool(x)
        >>> print(output.shape)  # (16, 128, 10, 16)

        >>> # With dilation (atrous pooling)
        >>> pool = MaxPool2d(3, stride=1, padding=1, dilation=2)
        >>> x = nova.randn(8, 64, 28, 28)
        >>> output = pool(x)
        >>> print(output.shape)  # (8, 64, 26, 26)

        >>> # ResNet-style architecture
        >>> class ResNetBlock(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = Conv2d(3, 64, 7, stride=2, padding=3)
        ...         self.pool = MaxPool2d(3, stride=2, padding=1)
        ...         self.layer1 = self._make_layer(64, 128)
        ...
        ...     def forward(self, x):
        ...         x = relu(self.conv1(x))
        ...         x = self.pool(x)  # Early aggressive downsampling
        ...         x = self.layer1(x)
        ...         return x

        >>> # AlexNet uses overlapping pooling
        >>> pool = MaxPool2d(3, stride=2)  # Overlapping windows

    Note:
        Max pooling is the standard choice in most modern architectures
        (ResNet, VGG, AlexNet, etc.) due to its effectiveness at:
        - Reducing spatial dimensions
        - Providing translation invariance
        - Preserving important features
    """

    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Optional[Stride] = None,
        padding: Padding = 0,
        dilation: Dilation = 1,
    ) -> None:
        super().__init__()

        self.KH, self.KW = _pair(kernel_size)
        self.SH, self.SW = _pair(stride) if stride is not None else (self.KH, self.KW)
        self.PH, self.PW = _pair(padding)
        self.DH, self.DW = _pair(dilation)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D max pooling.

        Args:
            input: Input tensor of shape (N, C, H, W).

        Returns:
            Pooled tensor of shape (N, C, H_out, W_out).
        """
        return F.max_pool2d(
            input,
            (self.KH, self.KW),
            (self.SH, self.SW),
            (self.PH, self.PW),
            (self.DH, self.DW),
        )

    def extra_repr(self) -> str:
        return "kernel_size=({KH}, {KW}), stride=({SH}, {SW}), padding=({PH}, {PW}), dilation=({DH}, {DW})".format(
            **self.__dict__
        )


class MaxPool3d(Module):
    """Applies 3D max pooling over the input volume.

    Computes the maximum value in 3D windows. Used for downsampling
    volumetric data in video understanding and 3D medical imaging.

    Args:
        kernel_size: Size of the pooling window. Can be int or tuple (D, H, W).
        stride: Stride of the pooling window. If None, defaults to kernel_size.
        padding: Zero-padding added to all sides. Default: 0
        dilation: Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`

    Examples::

        >>> # Downsample video uniformly
        >>> pool = MaxPool3d(kernel_size=2, stride=2)
        >>> x = nova.randn(8, 64, 16, 112, 112)  # (N, C, T, H, W)
        >>> output = pool(x)
        >>> print(output.shape)  # (8, 64, 8, 56, 56)

        >>> # Different pooling for temporal vs spatial
        >>> pool = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        >>> x = nova.randn(4, 128, 8, 56, 56)
        >>> output = pool(x)
        >>> print(output.shape)  # (4, 128, 8, 28, 28) - preserve temporal

        >>> # C3D network for action recognition
        >>> class C3D(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = Conv3d(3, 64, 3, padding=1)
        ...         self.pool1 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ...         self.conv2 = Conv3d(64, 128, 3, padding=1)
        ...         self.pool2 = MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ...
        ...     def forward(self, x):
        ...         x = relu(self.conv1(x))
        ...         x = self.pool1(x)  # Spatial pooling only
        ...         x = relu(self.conv2(x))
        ...         x = self.pool2(x)  # Spatiotemporal pooling
        ...         return x

        >>> # 3D medical imaging
        >>> x = nova.randn(2, 1, 64, 256, 256)  # CT scan volumes
        >>> pool = MaxPool3d(2, 2)
        >>> output = pool(x)
        >>> print(output.shape)  # (2, 1, 32, 128, 128)

    Note:
        In video tasks, it's common to use different kernel sizes for
        temporal vs spatial dimensions to preserve temporal resolution
        while reducing spatial dimensions.
    """

    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Optional[Stride] = None,
        padding: Padding = 0,
        dilation: Dilation = 1,
    ) -> None:
        super().__init__()
        self.KD, self.KH, self.KW = _triple(kernel_size)
        self.SD, self.SH, self.SW = (
            _triple(stride) if stride is not None else (self.KD, self.KH, self.KW)
        )
        self.PD, self.PH, self.PW = _triple(padding)
        self.DD, self.DH, self.DW = _triple(dilation)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D max pooling.

        Args:
            input: Input tensor of shape (N, C, D, H, W).

        Returns:
            Pooled tensor of shape (N, C, D_out, H_out, W_out).
        """
        return F.max_pool3d(
            input,
            (self.KD, self.KH, self.KW),
            (self.SD, self.SH, self.SW),
            (self.PD, self.PH, self.PW),
            (self.DD, self.DH, self.DW),
        )

    def extra_repr(self):
        return "kernel_size=({KD}, {KH}, {KW}), stride=({SD}, {SH}, {SW}), padding=({PD}, {PH}, {PW}), dilation=({DD},{DH},{DW})".format(
            **self.__dict__
        )
