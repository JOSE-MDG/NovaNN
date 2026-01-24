from __future__ import annotations
import nova
import math
import nova.nn.init as init
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module, LazyModuleMixin
from nova.nn.parameter import Parameter, UninitializedParameter
from nova.nn.utils import _single, _pair, _triple

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import KernelSize, Stride, Padding, PaddingMode, Dtype, Dilation


class Conv1d(Module):
    """Applies a 1D convolution over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, L)`
    and output :math:`(N, C_{out}, L_{out})` can be precisely described as:

    .. math::
        \\text{out}(N_i, C_{out_j}) = \\text{bias}(C_{out_j}) +
        \\sum_{k = 0}^{C_{in} - 1} \\text{weight}(C_{out_j}, k) \\star \\text{input}(N_i, k)

    where :math:`\\star` is the valid 1D cross-correlation operator, :math:`N` is the batch size,
    :math:`C` denotes the number of channels, and :math:`L` is the length of the signal sequence.

    Args:
        in_channels: Number of channels in the input signal
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Padding added to both sides of the input. Default: 0
        dilation: Spacing between kernel elements. Default: 1
        bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode: Padding mode to use. Options: ``'zeros'``, ``'reflect'``, ``'replicate'``,
            or ``'circular'``. Default: ``'zeros'``
        dtype: The desired data type of parameters. Default: None

    Attributes:
        weight: The learnable weights of the module of shape
            :math:`(\\text{out\\_channels}, \\text{in\\_channels}, \\text{kernel\\_size})`.
            The values are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`,
            where :math:`k = \\frac{1}{C_{in} * \\text{kernel\\_size}}`
        bias: The learnable bias of the module of shape :math:`(\\text{out\\_channels}, 1)`.
            If ``bias`` is ``True``, the values are initialized from
            :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where :math:`k = \\frac{1}{C_{in} * \\text{kernel\\_size}}`

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where

        .. math::
            L_{out} = \\left\\lfloor\\frac{L_{in} + 2 \\times \\text{padding} - \\text{dilation}
            \\times (\\text{kernel\\_size} - 1) - 1}{\\text{stride}} + 1\\right\\rfloor

    Examples::

        >>> # Standard 1D convolution
        >>> m = Conv1d(16, 33, 3, stride=2)
        >>> x = nova.randn(20, 16, 50)
        >>> output = m(x)
        >>> print(output.shape)
        (20, 33, 24)

        >>> # With padding
        >>> m = Conv1d(16, 33, 3, padding=1)
        >>> x = nova.randn(20, 16, 50)
        >>> output = m(x)
        >>> print(output.shape)
        (20, 33, 50)

        >>> # For time series data
        >>> m = Conv1d(1, 64, kernel_size=5)
        >>> signal = nova.randn(32, 1, 1000)  # 32 samples, 1 channel, 1000 timesteps
        >>> features = m(signal)
        >>> print(features.shape)
        (32, 64, 996)

        >>> # Without bias
        >>> m = Conv1d(16, 33, 3, bias=False)
        >>> print(m.bias)  # None
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize,
        stride: Stride = 1,
        padding: Padding = 0,
        dilation: Dilation = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = _single(kernel_size)
        self.S = _single(stride)
        self.P = _single(padding)
        self.D = _single(dilation)
        self.use_bias = bias
        self.padding_mode = padding_mode

        self.weight = Parameter(
            nova.empty((out_channels, in_channels, self.K)), dtype=dtype
        )
        if self.use_bias:
            self.bias = Parameter(nova.empty((out_channels, 1)), dtype=dtype)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets learnable parameters using Kaiming uniform initialization.

        Weight is initialized using Kaiming uniform initialization, and bias (if present)
        is initialized uniformly within a range based on the fan-in.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = init.get_fans(self.weight, mode="fan_in")
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 1D convolution to the input.

        Args:
            input: Input tensor of shape :math:`(N, C_{in}, L)` or :math:`(C_{in}, L)`

        Returns:
            Output tensor of shape :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`
        """
        return F.conv1d(
            input,
            self.weight,
            self.K,
            self.S,
            self.P,
            self.D,
            bias=self.bias,
            padding_mode=self.padding_mode,
        )

    def extra_repr(self):
        return "{in_channels}, {out_channels}, kernel_size={K}, stride={S}, padding={P}, bias={use_bias}".format(
            **self.__dict__
        )


class Conv2d(Module):
    """Applies a 2D convolution over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, H, W)`
    and output :math:`(N, C_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::
        \\text{out}(N_i, C_{out_j}) = \\text{bias}(C_{out_j}) +
        \\sum_{k = 0}^{C_{in} - 1} \\text{weight}(C_{out_j}, k) \\star \\text{input}(N_i, k)

    where :math:`\\star` is the valid 2D cross-correlation operator, :math:`N` is the batch size,
    :math:`C` denotes the number of channels, :math:`H` is the height of input planes in pixels,
    and :math:`W` is the width in pixels.

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel. Can be a single int or tuple (kH, kW)
        stride: Stride of the convolution. Can be a single int or tuple (sH, sW). Default: 1
        padding: Padding added to all four sides of the input. Can be a single int or tuple
            (padH, padW). Default: 0
        dilation: Spacing between kernel elements. Can be a single int or tuple (dH, dW). Default: 1
        bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode: Padding mode to use. Options: ``'zeros'``, ``'reflect'``, ``'replicate'``,
            or ``'circular'``. Default: ``'zeros'``
        dtype: The desired data type of parameters. Default: None

    Attributes:
        weight: The learnable weights of the module of shape
            :math:`(\\text{out\\_channels}, \\text{in\\_channels}, \\text{kernel\\_size[0]}, \\text{kernel\\_size[1]})`.
            The values are initialized using Kaiming uniform initialization
        bias: The learnable bias of the module of shape :math:`(\\text{out\\_channels}, 1)`.
            If ``bias`` is ``True``, initialized uniformly based on fan-in

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

        .. math::
            H_{out} = \\left\\lfloor\\frac{H_{in} + 2 \\times \\text{padding[0]} - \\text{dilation[0]}
            \\times (\\text{kernel\\_size[0]} - 1) - 1}{\\text{stride[0]}} + 1\\right\\rfloor

        .. math::
            W_{out} = \\left\\lfloor\\frac{W_{in} + 2 \\times \\text{padding[1]} - \\text{dilation[1]}
            \\times (\\text{kernel\\_size[1]} - 1) - 1}{\\text{stride[1]}} + 1\\right\\rfloor

    Examples::

        >>> # Standard 2D convolution with square kernels
        >>> m = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        >>> x = nova.randn(32, 3, 224, 224)
        >>> output = m(x)
        >>> print(output.shape)
        (32, 64, 224, 224)

        >>> # Non-square kernels and asymmetric padding
        >>> m = Conv2d(3, 33, kernel_size=(3, 5), padding=(1, 2))
        >>> x = nova.randn(20, 3, 50, 100)
        >>> output = m(x)
        >>> print(output.shape)
        (20, 33, 50, 100)

        >>> # Strided convolution (downsampling)
        >>> m = Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        >>> x = nova.randn(8, 16, 64, 64)
        >>> output = m(x)
        >>> print(output.shape)
        (8, 32, 32, 32)

        >>> # Typical CNN layer
        >>> conv = Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        >>> bn = BatchNorm2d(64)
        >>> relu = ReLU()
        >>> x = nova.randn(1, 3, 224, 224)
        >>> output = relu(bn(conv(x)))

        >>> # Without bias (common before BatchNorm)
        >>> m = Conv2d(64, 128, 3, padding=1, bias=False)
        >>> print(m.bias)  # None
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize,
        stride: Stride = 1,
        padding: Padding = 0,
        dilation: Dilation = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.KH, self.KW = _pair(kernel_size)
        self.SH, self.SW = _pair(stride)
        self.PH, self.PW = _pair(padding)
        self.DH, self.DW = _pair(dilation)
        self.use_bias: bool = bias
        self.padding_mode: PaddingMode = padding_mode

        self.weight = Parameter(
            nova.empty((out_channels, in_channels, self.KH, self.KW), dtype=dtype)
        )
        if self.use_bias:
            self.bias = Parameter(nova.empty((out_channels, 1), dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets learnable parameters using Kaiming uniform initialization.

        Weight is initialized using Kaiming uniform initialization, and bias (if present)
        is initialized uniformly within a range based on the fan-in.
        """
        init.kaiming_uniform_(self.weight)

        if self.use_bias:
            fan_in = init.get_fans(self.weight, mode="fan_in")
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 2D convolution to the input.

        Args:
            input: Input tensor of shape :math:`(N, C_{in}, H, W)` or :math:`(C_{in}, H, W)`

        Returns:
            Output tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`
        """
        return F.conv2d(
            input,
            self.weight,
            (self.KH, self.KW),
            (self.SH, self.SW),
            (self.PH, self.PW),
            (self.DH, self.DW),
            bias=self.bias,
            padding_mode=self.padding_mode,
        )

    def extra_repr(self):
        return "{in_channels}, {out_channels}, kernel_size=({KH}, {KW}), stride=({SH}, {SW}), padding=({PH}, {PW}), bias={use_bias}".format(
            **self.__dict__
        )


class Conv3d(Module):
    """Applies a 3D convolution over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::
        \\text{out}(N_i, C_{out_j}) = \\text{bias}(C_{out_j}) +
        \\sum_{k = 0}^{C_{in} - 1} \\text{weight}(C_{out_j}, k) \\star \\text{input}(N_i, k)

    where :math:`\\star` is the valid 3D cross-correlation operator, :math:`N` is the batch size,
    :math:`C` denotes the number of channels, :math:`D` is the depth of the volume, :math:`H` is
    the height, and :math:`W` is the width.

    This layer is commonly used for video processing, medical imaging (CT/MRI scans), and other
    volumetric data where spatial relationships exist in three dimensions.

    Args:
        in_channels: Number of channels in the input volume
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel. Can be a single int or tuple (kD, kH, kW)
        stride: Stride of the convolution. Can be a single int or tuple (sD, sH, sW). Default: 1
        padding: Padding added to all six sides of the input. Can be a single int or tuple
            (padD, padH, padW). Default: 0
        dilation: Spacing between kernel elements. Can be a single int or tuple (dD, dH, dW). Default: 1
        bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode: Padding mode to use. Options: ``'zeros'``, ``'reflect'``, ``'replicate'``,
            or ``'circular'``. Default: ``'zeros'``
        dtype: The desired data type of parameters. Default: None

    Attributes:
        weight: The learnable weights of the module of shape
            :math:`(\\text{out\\_channels}, \\text{in\\_channels}, \\text{kernel\\_size[0]}, \\text{kernel\\_size[1]}, \\text{kernel\\_size[2]})`.
            The values are initialized using Kaiming uniform initialization
        bias: The learnable bias of the module of shape :math:`(\\text{out\\_channels}, 1)`.
            If ``bias`` is ``True``, initialized uniformly based on fan-in

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` or :math:`(C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or :math:`(C_{out}, D_{out}, H_{out}, W_{out})`, where

        .. math::
            D_{out} = \\left\\lfloor\\frac{D_{in} + 2 \\times \\text{padding[0]} - \\text{dilation[0]}
            \\times (\\text{kernel\\_size[0]} - 1) - 1}{\\text{stride[0]}} + 1\\right\\rfloor

        .. math::
            H_{out} = \\left\\lfloor\\frac{H_{in} + 2 \\times \\text{padding[1]} - \\text{dilation[1]}
            \\times (\\text{kernel\\_size[1]} - 1) - 1}{\\text{stride[1]}} + 1\\right\\rfloor

        .. math::
            W_{out} = \\left\\lfloor\\frac{W_{in} + 2 \\times \\text{padding[2]} - \\text{dilation[2]}
            \\times (\\text{kernel\\_size[2]} - 1) - 1}{\\text{stride[2]}} + 1\\right\\rfloor

    Examples::

        >>> # Standard 3D convolution for video processing
        >>> m = Conv3d(3, 64, kernel_size=3, padding=1)
        >>> video = nova.randn(8, 3, 16, 112, 112)  # 8 videos, 3 channels (RGB), 16 frames
        >>> output = m(video)
        >>> print(output.shape)
        (8, 64, 16, 112, 112)

        >>> # With stride for temporal downsampling
        >>> m = Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        >>> x = nova.randn(4, 64, 16, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        (4, 128, 8, 16, 16)

        >>> # Medical imaging (CT/MRI volumes)
        >>> m = Conv3d(1, 32, kernel_size=3, padding=1)
        >>> ct_scan = nova.randn(2, 1, 64, 64, 64)  # 2 volumes, 1 channel
        >>> features = m(ct_scan)
        >>> print(features.shape)
        (2, 32, 64, 64, 64)

        >>> # Non-cubic kernels for anisotropic data
        >>> m = Conv3d(16, 32, kernel_size=(1, 3, 3))  # Small temporal, larger spatial
        >>> x = nova.randn(1, 16, 32, 128, 128)
        >>> output = m(x)
        >>> print(output.shape)
        (1, 32, 32, 126, 126)

        >>> # Without bias (common before BatchNorm3d)
        >>> m = Conv3d(64, 128, 3, padding=1, bias=False)
        >>> print(m.bias)  # None
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize,
        stride: Stride = 1,
        padding: Padding = 0,
        dilation: Dilation = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KD, self.KH, self.KW = _triple(kernel_size)
        self.SD, self.SH, self.SW = _triple(stride)
        self.PD, self.PH, self.PW = _triple(padding)
        self.DD, self.DH, self.DW = _triple(dilation)
        self.use_bias: bool = bias
        self.padding_mode = padding_mode

        self.weight = Parameter(
            nova.empty(
                (out_channels, in_channels, self.KD, self.KH, self.KW), dtype=dtype
            )
        )

        if self.use_bias:
            self.bias = Parameter(nova.empty((out_channels, 1), dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets learnable parameters using Kaiming uniform initialization.

        Weight is initialized using Kaiming uniform initialization, and bias (if present)
        is initialized uniformly within a range based on the fan-in.
        """
        init.kaiming_uniform_(self.weight)

        if self.use_bias:
            fan_in = init.get_fans(self.weight, mode="fan_in")
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """Applies 3D convolution to the input.

        Args:
            input: Input tensor of shape :math:`(N, C_{in}, D, H, W)` or :math:`(C_{in}, D, H, W)`

        Returns:
            Output tensor of shape :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or
            :math:`(C_{out}, D_{out}, H_{out}, W_{out})`
        """
        return F.conv3d(
            input,
            self.weight,
            (self.KD, self.KH, self.KW),
            (self.SD, self.SH, self.SW),
            (self.PD, self.PH, self.PW),
            (self.DD, self.DH, self.DW),
            bias=self.bias,
            padding_mode=self.padding_mode,
        )

    def extra_repr(self) -> str:
        return "{in_channels}, {out_channels}, kernel_size=({KD}, {KH}, {KW}), stride=({SD}, {SH}, {SW}), padding=({PD}, {PH}, {PW}), bias={use_bias}".format(
            **self.__dict__
        )


class _LazyConvXdMixin(LazyModuleMixin):
    """Base mixin class for lazy convolutional layers.

    This internal class provides lazy initialization functionality for convolutional
    layers. The number of input channels is automatically inferred from the first
    forward pass, allowing for more flexible model construction.

    Args:
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Padding added to the input. Default: 0
        dilation: Spacing between kernel elements. Default: 1
        bias: If ``True``, adds a learnable bias. Default: ``True``
        padding_mode: Padding mode to use. Default: ``'zeros'``
        dtype: The desired data type of parameters. Default: None

    Attributes:
        weight: Uninitialized learnable weight parameter (initialized on first forward)
        bias: Uninitialized learnable bias parameter (initialized on first forward)
    """

    in_channels: int
    out_channels: int
    weight: UninitializedParameter
    bias: Optional[UninitializedParameter]

    def __init__(
        self,
        out_channels: int,
        kernel_size: KernelSize,
        stride: Stride = 1,
        padding: Padding = 0,
        dilation: Dilation = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        dtype: Optional[Dtype] = None,
    ) -> None:
        Module.__init__(self)

        self.out_channels = out_channels
        self.use_bias = bias
        self.padding_mode = padding_mode
        self.dtype = dtype

        self._store_attributes(kernel_size, stride, padding, dilation)

        self.weight = UninitializedParameter()
        if bias:
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("bias", None)

    def _store_attributes(
        self,
        kernel_size: KernelSize,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
    ):
        """Stores kernel attributes in the appropriate format for the subclass.

        Must be overridden by each subclass to store kernel_size, stride, padding,
        and dilation in the format appropriate for that dimensionality.

        Args:
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to the input
            dilation: Spacing between kernel elements

        Raises:
            NotImplementedError: Always raised as this must be implemented by subclasses
        """
        raise NotImplementedError

    def reset_parameters(self) -> None:
        """Resets parameters if they have been initialized.

        This method only resets parameters after lazy initialization has occurred.
        Before initialization, this method does nothing.
        """
        if not self.has_uninitialized_params():
            super().reset_parameters()

    def initialize_parameters(self, input: Tensor) -> None:
        """Infers input channels from input and initializes all parameters.

        This method is called automatically on the first forward pass. It infers
        ``in_channels`` from the channel dimension of the input tensor.

        Args:
            input: Input tensor used to infer the number of input channels
        """
        if self.has_uninitialized_params():
            with nova.no_grad():
                self.in_channels = self._get_in_channels(input)

                weight_shape = self._get_weight_shape()

                self.weight = self.weight.materialize(weight_shape, dtype=self.dtype)

                if self.use_bias:
                    self.bias = self.bias.materialize(
                        (self.out_channels, 1), dtype=self.dtype
                    )

                self.reset_parameters()

    def _get_weight_shape(self) -> tuple:
        """Returns the weight shape according to the layer's dimensionality.

        Must be implemented by each subclass to return the appropriate weight shape.

        Returns:
            Tuple representing the shape of the weight tensor

        Raises:
            NotImplementedError: Always raised as this must be implemented by subclasses
        """
        raise NotImplementedError

    def _get_in_channels(self, input: Tensor) -> int:
        """Extracts the number of input channels from the input tensor.

        Args:
            input: Input tensor

        Returns:
            Number of input channels

        Raises:
            RuntimeError: If input dimensions don't match expected dimensions for this layer
        """
        num_spatial_dims = self._get_num_spatial_dims()
        num_dims_no_batch = num_spatial_dims + 1
        num_dims_batch = num_dims_no_batch + 1
        if input.dim() not in (num_dims_no_batch, num_dims_batch):
            raise RuntimeError(
                f"Expected {num_dims_no_batch}D (unbatched) or {num_dims_batch}D (batched) input "
                f"to {self.__class__.__name__}, but "
                f"got input of size: {input.shape}"
            )
        return input.shape[1] if input.dim() == num_dims_batch else input.shape[0]

    def _get_num_spatial_dims(self) -> int:
        """Returns the number of spatial dimensions for this convolution type.

        Returns:
            Number of spatial dimensions (1 for Conv1d, 2 for Conv2d, 3 for Conv3d)

        Raises:
            NotImplementedError: Always raised as this must be implemented by subclasses
        """
        raise NotImplementedError


class LazyConv1d(_LazyConvXdMixin, Conv1d):
    """A Conv1d layer with lazy initialization of the ``in_channels`` argument.

    The number of input channels is inferred from the ``input.shape[1]`` on the first
    forward pass. This is useful when building models where the number of input channels
    is unknown at construction time.

    Args:
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Padding added to both sides of the input. Default: 0
        dilation: Spacing between kernel elements. Default: 1
        bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode: Padding mode to use. Default: ``'zeros'``
        dtype: The desired data type of parameters. Default: None

    Shape:
        - Input: :math:`(N, C_{in}, L)` or :math:`(C_{in}, L)`
        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`

    Examples::

        >>> # in_channels is inferred on first forward
        >>> m = LazyConv1d(33, kernel_size=3, stride=2)
        >>> x = nova.randn(20, 16, 50)
        >>> output = m(x)
        >>> print(m.in_channels)  # 16 (inferred from input)
        >>> print(output.shape)
        (20, 33, 24)

        >>> # Useful in sequential models with unknown intermediate shapes
        >>> model = Sequential(
        ...     LazyConv1d(64, 3),
        ...     ReLU(),
        ...     LazyConv1d(128, 3),  # Automatically adapts to 64 input channels
        ...     ReLU()
        ... )
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: KernelSize,
        stride: Stride = 1,
        padding: Padding = 0,
        dilation: Dilation = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        dtype: Optional[Dtype] = None,
    ) -> None:
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        super().__init__(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            dtype=dtype,
        )

    def _store_attributes(
        self,
        kernel_size: KernelSize,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
    ):
        """Stores 1D convolution attributes."""
        self.K = kernel_size
        self.S = stride
        self.P = padding
        self.D = dilation

    def _get_weight_shape(self) -> tuple:
        """Returns the weight shape for 1D convolution.

        Returns:
            Tuple of shape (out_channels, in_channels, kernel_size)
        """
        return (self.out_channels, self.in_channels, self.K)

    def _get_num_spatial_dims(self) -> int:
        """Returns 1 for 1D convolution."""
        return 1


class LazyConv2d(_LazyConvXdMixin, Conv2d):
    """A Conv2d layer with lazy initialization of the ``in_channels`` argument.

    The number of input channels is inferred from the ``input.shape[1]`` on the first
    forward pass. This is particularly useful when building dynamic CNN architectures
    where the number of input channels is determined at runtime.

    Args:
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Padding added to all sides of the input. Default: 0
        dilation: Spacing between kernel elements. Default: 1
        bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode: Padding mode to use. Default: ``'zeros'``
        dtype: The desired data type of parameters. Default: None

    Shape:
        - Input: :math:`(N, C_{in}, H, W)` or :math:`(C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`

    Examples::

        >>> # in_channels is inferred on first forward
        >>> m = LazyConv2d(64, kernel_size=3, padding=1)
        >>> x = nova.randn(32, 3, 224, 224)
        >>> output = m(x)
        >>> print(m.in_channels)  # 3 (inferred from input)
        >>> print(output.shape)
        (32, 64, 224, 224)

        >>> # Building a CNN without specifying intermediate channels
        >>> model = Sequential(
        ...     LazyConv2d(64, 7, stride=2, padding=3),
        ...     LazyBatchNorm2d(),
        ...     ReLU(),
        ...     LazyConv2d(128, 3, padding=1),  # Adapts to 64 channels
        ...     LazyBatchNorm2d(),
        ...     ReLU()
        ... )
        >>> x = nova.randn(1, 3, 224, 224)
        >>> output = model(x)
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: KernelSize,
        stride: Stride = 1,
        padding: Padding = 0,
        dilation: Dilation = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        dtype: Optional[Dtype] = None,
    ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            dtype=dtype,
        )

    def _store_attributes(
        self,
        kernel_size: KernelSize,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
    ):
        """Stores 2D convolution attributes."""
        self.KH, self.KW = kernel_size
        self.SH, self.SW = stride
        self.PH, self.PW = padding
        self.DH, self.DW = dilation

    def _get_weight_shape(self) -> tuple:
        """Returns the weight shape for 2D convolution.

        Returns:
            Tuple of shape (out_channels, in_channels, kernel_height, kernel_width)
        """
        return (self.out_channels, self.in_channels, self.KH, self.KW)

    def _get_num_spatial_dims(self) -> int:
        """Returns 2 for 2D convolution."""
        return 2


class LazyConv3d(_LazyConvXdMixin, Conv3d):
    """A Conv3d layer with lazy initialization of the ``in_channels`` argument.

    The number of input channels is inferred from the ``input.shape[1]`` on the first
    forward pass. This is useful for building 3D architectures where the number of
    input channels may be determined dynamically, particularly in video processing
    or medical imaging pipelines.

    Args:
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel. Can be a single int or tuple (kD, kH, kW)
        stride: Stride of the convolution. Default: 1
        padding: Padding added to all sides of the input. Default: 0
        dilation: Spacing between kernel elements. Default: 1
        bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode: Padding mode to use. Default: ``'zeros'``
        dtype: The desired data type of parameters. Default: None

    Shape:
        - Input: :math:`(N, C_{in}, D, H, W)` or :math:`(C_{in}, D, H, W)`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or :math:`(C_{out}, D_{out}, H_{out}, W_{out})`

    Examples::

        >>> # in_channels is inferred on first forward
        >>> m = LazyConv3d(64, kernel_size=3, padding=1)
        >>> video = nova.randn(8, 3, 16, 112, 112)
        >>> output = m(video)
        >>> print(m.in_channels)  # 3 (inferred from input)
        >>> print(output.shape)
        (8, 64, 16, 112, 112)

        >>> # Building a 3D CNN without specifying intermediate channels
        >>> model = Sequential(
        ...     LazyConv3d(64, kernel_size=3, padding=1),
        ...     LazyBatchNorm3d(),
        ...     ReLU(),
        ...     LazyConv3d(128, kernel_size=3, stride=2, padding=1),  # Adapts to 64 channels
        ...     LazyBatchNorm3d(),
        ...     ReLU(),
        ...     LazyConv3d(256, kernel_size=3, padding=1)  # Adapts to 128 channels
        ... )

        >>> # For medical imaging with unknown input modalities
        >>> m = LazyConv3d(32, kernel_size=3, padding=1)
        >>> # Works with single-channel CT
        >>> ct = nova.randn(1, 1, 64, 64, 64)
        >>> out1 = m(ct)
        >>> # Or multi-channel MRI after reinitialization
        >>> m = LazyConv3d(32, kernel_size=3, padding=1)
        >>> mri = nova.randn(1, 4, 64, 64, 64)
        >>> out2 = m(mri)

        >>> # Video processing with temporal downsampling
        >>> m = LazyConv3d(128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        >>> x = nova.randn(2, 64, 32, 64, 64)
        >>> output = m(x)
        >>> print(output.shape)
        (2, 128, 16, 32, 32)
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: KernelSize,
        stride: Stride = 1,
        padding: Padding = 0,
        dilation: Dilation = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
        dtype: Optional[Dtype] = None,
    ) -> None:
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super().__init__(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            dtype=dtype,
        )

    def _store_attributes(
        self,
        kernel_size: KernelSize,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
    ):
        """Stores 3D convolution attributes."""
        self.KD, self.KH, self.KW = kernel_size
        self.SD, self.SH, self.SW = stride
        self.PD, self.PH, self.PW = padding
        self.DD, self.DH, self.DW = dilation

    def _get_weight_shape(self) -> tuple:
        """Returns the weight shape for 3D convolution.

        Returns:
            Tuple of shape (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
        """
        return (self.out_channels, self.in_channels, self.KD, self.KH, self.KW)

    def _get_num_spatial_dims(self) -> int:
        """Returns 3 for 3D convolution."""
        return 3
