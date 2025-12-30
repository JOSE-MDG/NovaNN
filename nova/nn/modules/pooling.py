from __future__ import annotations
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import KernelSize, Padding, Stride, Dilation


def _pair(input: int | tuple[int, int]) -> tuple[int, int]:

    if input is None:
        return None

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

    if input is None:
        return None

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


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size: Optional[int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool1d(input, self.output_size)

    def extra_repr(self):
        return f"output_size={self.output_size}"


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: tuple[Optional[int], Optional[int]]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(input, self.output_size)

    def extra_repr(self):
        return f"output_size={self.output_size}"


class AdaptiveAvgPool3d(Module):
    def __init__(
        self, output_size: tuple[Optional[int], Optional[int], Optional[int]]
    ) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool3d(input, self.output_size)

    def extra_repr(self):
        return f"output_size={self.output_size}"


class AvgPool1d(Module):
    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Optional[Stride] = None,
        padding: Padding = 0,
    ) -> None:
        super().__init__()
        self.K = kernel_size
        self.S = stride
        self.P = padding

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool1d(input, self.K, self.S, self.P)

    def extra_repr(self):
        return "kernel_size={K}, stride={S}, padding={P}".format(**self.__dict__)


class AvgPool2d(Module):
    def __init__(
        self, kernel_size: KernelSize, stride: Stride = None, padding: Padding = 0
    ) -> None:
        super().__init__()
        self.KH, self.KW = _pair(kernel_size)
        self.SH, self.SW = _pair(stride)
        self.PH, self.PW = _pair(padding)

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool2d(
            input, (self.KH, self.KW), (self.SH, self.SW), (self.PH, self.PW)
        )

    def extra_repr(self):
        return "kernel_size=({KH}, {KW}), stride=({SH}, {SW}), padding=({PH}, {PW})".format(
            **self.__dict__
        )


class AvgPool3d(Module):
    def __init__(
        self, kernel_size: KernelSize, stride: Stride = None, padding: Padding = 0
    ) -> None:
        super().__init__()
        self.KD, self.KH, self.KW = _triple(kernel_size)
        self.SD, self.SH, self.SW = _triple(stride)
        self.PD, self.PH, self.PW = _triple(padding)

    def forward(self, input: Tensor) -> Tensor:
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
    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Optional[Stride] = 1,
        padding: Padding = 0,
        dilation: Dilation = 1,
    ) -> None:
        super().__init__()
        self.K = kernel_size
        self.S = stride
        self.P = padding
        self.D = dilation

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool1d(input, self.K, self.S, self.P, self.D)

    def extra_repr(self) -> str:
        return "kernel_size={K}, stride={S}, padding={P}, dilation={D}".format(
            **self.__dict__
        )


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Optional[Stride] = None,
        padding: Padding = 0,
        dilation: Dilation = 1,
    ) -> None:
        super().__init__()

        self.KH, self.KW = _pair(kernel_size)
        self.SH, self.SW = _pair(stride)
        self.PH, self.PW = _pair(padding)
        self.DH, self.DW = _pair(dilation)

    def forward(self, input: Tensor) -> Tensor:
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
    def __init__(
        self,
        kernel_size: KernelSize,
        stride: Optional[Stride] = None,
        padding: Padding = 0,
        dilation: Dilation = 1,
    ) -> None:
        super().__init__()
        self.KD, self.KH, self.KW = _triple(kernel_size)
        self.SD, self.SH, self.SW = _triple(stride)
        self.PD, self.PH, self.PW = _triple(padding)
        self.DD, self.DH, self.DW = _triple(dilation)

    def forward(self, input: Tensor) -> Tensor:
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
