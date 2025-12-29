from __future__ import annotations
import nova
import math
import nova.nn.init as init
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import KernelSize, Stride, Padding, PaddingMode, Dtype, Dilation


def _pair(input: int | tuple[int, int]) -> tuple[int, int]:

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
        self.K = kernel_size
        self.S = stride
        self.P = padding
        self.D = dilation
        self.use_bias = bias
        self.padding_mode = padding_mode

        self.weight: Parameter = Parameter(
            nova.empty((out_channels, in_channels, self.K), dtype=dtype)
        )
        if self.use_bias:
            self.bias: Parameter = Parameter(nova.empty((out_channels, 1), dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = init.get_fans(self.weight.size, mode="fan_in")
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

        self.weight: Parameter = Parameter(
            nova.empty((out_channels, in_channels, self.KH, self.KW), dtype=dtype)
        )
        if self.use_bias:
            self.bias: Parameter = Parameter(nova.empty((out_channels, 1), dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = init.get_fans(self.weight.size, mode="fan_in")
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
        super.__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KD, self.KH, self.KW = _triple(kernel_size)
        self.SD, self.SH, self.SW = _triple(stride)
        self.PD, self.PH, self.PW = _triple(padding)
        self.DD, self.DH, self.DW = _triple(dilation)
        self.use_bias: bool = bias
        self.padding_mode = padding_mode

        self.weight: Parameter = Parameter(
            nova.empty(
                (out_channels, in_channels, self.KD, self.KH, self.KW), dtype=dtype
            )
        )

        if self.use_bias:
            self.bias: Parameter = Parameter(nova.empty((out_channels, 1), dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:

            fan_in = init.get_fans(self.weight.size, mode="fan_in")
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
