from __future__ import annotations
import nova
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova._typing import KernelSize, Padding, Dilation, Stride, PaddingMode
    from nova import Tensor


def _single(input: int | tuple[int, int] | str) -> int:
    """
    Ensures the given input is a integer.

    Used internally by 1D operations (e.g., Conv1d, AvgPool1d) to handle
    parameters that can be provided as a pair or a tuple.

    Args:
        input: Integer, tuple of two integers or a string.

    Returns:
        A integer.

    Raises:
        ValueError: If a string like "same" is provided (not supported yet).
    """
    if isinstance(input, tuple):
        return input[0]

    elif isinstance(input, str):
        if input == "valid":
            return 0
        elif input == "same":
            raise ValueError("The 'same' value is not currently supported")
        else:
            raise ValueError(f"Unsupported value '{input}'")

    return int(input)


def _pair(input: int | tuple[int, int]) -> tuple[int, int]:
    """
    Ensures the given input is a tuple of two integers.

    Used internally by 2D operations (e.g., Conv2d, AvgPool2d) to handle
    parameters that can be provided as a single int or a tuple.

    Args:
        input: Integer, tuple of two integers or a string.

    Returns:
        Tuple of two integers.

    Raises:
        ValueError: If a string like "same" is provided (not supported yet).
    """
    if isinstance(input, int):
        return (input, input)

    elif isinstance(input, str):
        if input == "valid":
            return (0, 0)
        elif input == "same":
            raise ValueError("The 'same' value is not currently supported")
        else:
            raise ValueError(f"Unsupported value '{input}'")

    return tuple(input)


def _triple(input: int | tuple[int, int, int] | str) -> tuple[int, int, int]:
    """
    Ensures the given input is a tuple of three integers.

    Used internally by 3D operations (e.g., Conv3d, AvgPool3d) to handle
    parameters that can be provided as a single int or a tuple.

    Args:
        input: Integer, tuple of three integers or a string.

    Returns:
        Tuple of three integers.

    Raises:
        ValueError: If a string like "same" is provided (not supported yet).
    """
    if isinstance(input, int):
        return (input, input, input)

    elif isinstance(input, str):
        if input == "valid":
            return (0, 0, 0)
        elif input == "same":
            raise ValueError("The 'same' value is not currently supported")
        else:
            raise ValueError(f"Unsupported value '{input}'")
    return tuple(input)


def add_padding(input: Tensor, padding: Padding, padding_mode: PaddingMode) -> Tensor:
    """
    Adds padding to the input tensor based on its dimensionality.

    Used internally by convolution and pooling operations to apply padding
    to 1D, 2D, or 3D tensors. The padding is applied symmetrically on both
    sides of each spatial dimension.

    Args:
        input: Input tensor of shape (N, C, L) for 1D, (N, C, H, W) for 2D,
               or (N, C, D, H, W) for 3D.
        padding: Padding size(s). Integer for 1D, tuple of two integers for 2D,
                 or tuple of three integers for 3D.
        padding_mode: Padding mode to use (e.g., 'constant', 'reflect', 'replicate').

    Returns:
        Padded tensor with the same number of dimensions as input.

    Raises:
        ValueError: Raise ValueError if input dimensionality is not 3, 4, or 5.
    """
    if input.dim() == 3:
        P = padding
        pad_width = ((0, 0), (0, 0), (P, P))
    elif input.dim() == 4:
        PH, PW = padding
        pad_width = ((0, 0), (0, 0), (PH, PH), (PW, PW))
    else:
        PD, PH, PW = padding
        pad_width = ((0, 0), (0, 0), (PD, PD), (PH, PH), (PW, PW))

    return nova.pad(input, pad_width, padding_mode)


def calculate_out_size(
    *args: int,
    kernel_size: KernelSize,
    padding: Padding,
    stride: Stride,
    dilation: Optional[Dilation] = None,
) -> int | tuple[int, int] | tuple[int, int, int]:
    """
    Calculates the output spatial dimensions after a convolution or pooling operation.

    Used internally to compute the output size based on input dimensions, kernel size,
    padding, stride, and optionally dilation. Supports 1D, 2D, and 3D operations.

    Args:
        *args: Input spatial dimensions. Single integer for 1D (L), two integers
               for 2D (H, W), or three integers for 3D (D, H, W).
        kernel_size: Size of the kernel. Integer for 1D, tuple of two integers for 2D,
                     or tuple of three integers for 3D.
        padding: Padding applied to input. Integer for 1D, tuple of two integers for 2D,
                 or tuple of three integers for 3D.
        stride: Stride of the operation. Integer for 1D, tuple of two integers for 2D,
                or tuple of three integers for 3D.
        dilation: Dilation factor (optional). Integer for 1D, tuple of two integers for 2D,
                  or tuple of three integers for 3D. Defaults to None (no dilation).

    Returns:
        Output spatial dimension(s). Single integer for 1D, tuple of two integers for 2D,
        or tuple of three integers for 3D.

    Raises:
        ValueError: If more than 3 arguments are provided.
    """
    if len(args) > 3:
        raise ValueError("Many arguments to process")

    if len(args) == 1:
        L = args[0]

        K = kernel_size
        P = padding
        S = stride

        if dilation is not None:
            D = dilation
            K_eff = (K - 1) * D + 1
        else:
            K_eff = K

        L_out = (L + 2 * P - K_eff) // S + 1
        return L_out

    if len(args) == 2:
        H = args[0]
        W = args[1]

        KH, KW = kernel_size
        PH, PW = padding
        SH, SW = stride

        if dilation is not None:
            DH, DW = dilation
            KH_eff = (KH - 1) * DH + 1
            KW_eff = (KW - 1) * DW + 1
        else:
            KH_eff, KW_eff = KH, KW

        H_out = (H + 2 * PH - KH_eff) // SH + 1
        W_out = (W + 2 * PW - KW_eff) // SW + 1

        return H_out, W_out

    if len(args) == 3:
        D = args[0]
        H = args[1]
        W = args[2]

        KD, KH, KW = kernel_size
        PD, PH, PW = padding
        SD, SH, SW = stride

        if dilation is not None:
            DD, DH, DW = dilation
            KD_eff = (KD - 1) * DD + 1
            KH_eff = (KH - 1) * DH + 1
            KW_eff = (KW - 1) * DW + 1
        else:
            KD_eff, KH_eff, KW_eff = KD, KH, KW

        D_out = (D + 2 * PD - KD_eff) // SD + 1
        H_out = (H + 2 * PH - KH_eff) // SH + 1
        W_out = (W + 2 * PW - KW_eff) // SW + 1

        return D_out, H_out, W_out
