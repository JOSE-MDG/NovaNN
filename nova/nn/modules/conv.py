from __future__ import annotations
import nova
import math
import nova.nn.init as init
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module, LazyModuleMixin
from nova.nn.parameter import Parameter, UninitializedParameter

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import KernelSize, Stride, Padding, PaddingMode, Dtype, Dilation


def _single(input: int | tuple[int, int] | str) -> int:

    if isinstance(input, tuple):
        return input[0]

    elif isinstance(input, str):
        if input == "valid":
            return 0
        elif input == "same":
            raise ValueError(f"The 'same' value is not currently supported")
        else:
            raise ValueError(f"Unsupported value '{input}'")

    return int(input)


def _pair(input: int | tuple[int, int] | str) -> tuple[int, int]:

    if isinstance(input, int):
        return (input, input)

    elif isinstance(input, str):
        if input == "valid":
            return (0, 0)
        elif input == "same":
            raise ValueError(f"The 'same' value is not currently supported")
        else:
            raise ValueError(f"Unsupported value '{input}'")

    return tuple(input)


def _triple(input: int | tuple[int, int, int] | str) -> tuple[int, int, int]:

    if isinstance(input, int):
        return (input, input, input)

    elif isinstance(input, str):
        if input == "valid":
            return (0, 0, 0)
        elif input == "same":
            raise ValueError(f"The 'same' value is not currently supported")
        else:
            raise ValueError(f"Unsupported value '{input}'")
    return tuple(input)


class Conv1d(Module):

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

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = init.get_fans(self.weight, mode="fan_in")
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

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

        init.kaiming_uniform_(
            self.weight,
        )

        if self.use_bias:
            fan_in = init.get_fans(self.weight, mode="fan_in")
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

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

        init.kaiming_uniform_(
            self.weight,
        )

        if self.use_bias:
            fan_in = init.get_fans(self.weight, mode="fan_in")
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

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
        return "{in_channels}, {out_channels}, kernel_size=({KD}, {KH}, {KW}), stride=({SD}, {SH}, {SW}), padding=({PD}, {PH}, {PW}), bias={use_bias}"


class _LazyConvXdMixin(LazyModuleMixin):

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
        """Overwritten by each subclass to save in its format"""
        raise NotImplementedError

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params():
            super().reset_parameters()

    def initialize_parameters(self, input: Tensor) -> None:
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
        """The weight shape returns according to the dimensionality"""
        raise NotImplementedError

    def _get_in_channels(self, input: Tensor) -> int:
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
        raise NotImplementedError


class LazyConv1d(_LazyConvXdMixin, Conv1d):

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
        self.K = kernel_size
        self.S = stride
        self.P = padding
        self.D = dilation

    def _get_weight_shape(self) -> tuple:
        return (self.out_channels, self.in_channels, self.K)

    def _get_num_spatial_dims(self) -> int:
        return 1


class LazyConv2d(_LazyConvXdMixin, Conv2d):

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
        self.KH, self.KW = kernel_size
        self.SH, self.SW = stride
        self.PH, self.PW = padding
        self.DH, self.DW = dilation

    def _get_weight_shape(self) -> tuple:
        return (self.out_channels, self.in_channels, self.KH, self.KW)

    def _get_num_spatial_dims(self) -> int:
        return 2


class LazyConv3d(_LazyConvXdMixin, Conv3d):

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
        self.KD, self.KH, self.KW = kernel_size
        self.SD, self.SH, self.SW = stride
        self.PD, self.PH, self.PW = padding
        self.DD, self.DH, self.DW = dilation

    def _get_weight_shape(self) -> tuple:
        return (self.out_channels, self.in_channels, self.KD, self.KH, self.KW)

    def _get_num_spatial_dims(self) -> int:
        return 3
