from __future__ import annotations
import nova
import math
from typing import TYPE_CHECKING, Literal, Optional
from nova.utils import ensure_tensor

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dim, KernelSize, Stride, Padding
    from nova.nn import Parameter, Buffer


# Activations


def relu(input: Tensor) -> Tensor:
    input = ensure_tensor(input)

    return nova.maximum(0, input)


def leaky_relu(input: Tensor, alpha: float = 0.01) -> Tensor:
    input = ensure_tensor(input)
    alpha = ensure_tensor(alpha)

    return nova.where(input > 0, input, input * alpha)


def gelu(input: Tensor) -> Tensor:
    input = ensure_tensor(input)
    inner = math.sqrt(2.0 / math.pi) * (input + 0.044715 * nova.pow(input, 3))
    return 0.5 * input * (1.0 + nova.tanh(inner))


def prelu(input: Tensor, weight: float = 0.25) -> Tensor:
    input = ensure_tensor(input)
    weight = ensure_tensor(weight)

    return nova.maximum(0, input) + weight * nova.minimum(0, input)


def sigmoid(input: Tensor) -> Tensor:
    input = ensure_tensor(input)
    return nova.where(
        input >= 0, 1 / (1 + nova.exp(-input)), nova.exp(input) / (1 + nova.exp(input))
    )


def tanh(input: Tensor) -> Tensor:
    input = ensure_tensor(input)

    return input.tanh()


def softmax(input: Tensor, dim: Dim = 1) -> Tensor:

    logits = ensure_tensor(input)

    stable_logits = logits - nova.max(logits, dim=dim, keepdims=True)
    exp_logits = nova.exp(stable_logits)
    sum_exp = nova.sum(exp_logits, dim=dim, keepdims=True)
    return exp_logits / sum_exp


def log_softmax(input: Tensor, dim: Dim = 1) -> Tensor:
    input = ensure_tensor(input)

    M = nova.max(input, dim=dim, keepdims=True)
    sum_exp = nova.sum(nova.exp(input - M), dim=dim, keepdims=True)
    return (input - M) - nova.log(sum_exp)


# criterion
def mse_loss(input: Tensor, target: Tensor, weight: Optional[Tensor] = None) -> Tensor:

    input = ensure_tensor(input)
    target = ensure_tensor(target)

    loss = (target - input) ** 2

    if weight is not None:

        weight = ensure_tensor(weight)

        if weight.size != target.size:
            raise ValueError(
                f"weights and targets must be have the same shape, {weight.size} != {target.size}"
            )

        loss = loss * weight

    return nova.mean(loss)


def l1_loss(input: Tensor, target: Tensor, weight: Optional[Tensor] = None) -> Tensor:

    logits = ensure_tensor(input)

    loss = nova.abs(target - logits)

    if weight is not None:

        weight = ensure_tensor(weight)

        if weight.size != target.size:
            raise ValueError(
                f"weights and targets must be have the same shape, {weight.size} != {target.size}"
            )

        loss = loss * weight

    return nova.mean(loss)


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    beta: float = 1.0,
    reduction: Literal["none", "mean", "sum"] = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor:

    logits = ensure_tensor(input)
    target = ensure_tensor(target)

    diff = nova.abs(target - logits)

    loss = nova.where(diff < beta, 0.5 * (diff**2) / beta, diff - 0.5 * beta)

    if weight is not None:
        weight = ensure_tensor(weight)

        if weight.size != target.size:
            raise ValueError(
                f"weights and targets must be have the same shape, {weight.size} != {target.size}"
            )

        loss = loss * weight

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()

    else:
        raise ValueError(f"reduction expect ('sum','mean','none'), got '{reduction}'")


def binary_cross_entropy(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None
) -> Tensor:
    input = ensure_tensor(input)
    target = ensure_tensor(target)

    loss = -(target * nova.log(input) + (1 - target) * nova.log(1 - input))

    if weight is not None:
        weight = ensure_tensor(weight)

        if weight.size != target.size:
            raise ValueError(
                f"weights and targets must be have the same shape, {weight.size} != {target.size}"
            )

        loss = loss * weight[target]

    return nova.mean(loss)


def binary_cross_entropy_with_logits(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None
) -> Tensor:
    logits = ensure_tensor(input)
    target = ensure_tensor(target)

    loss = (
        nova.maximum(logits, 0)
        - logits * target
        + nova.log(1 + nova.exp(-nova.abs(logits)))
    )

    if weight is not None:
        weight = ensure_tensor(weight)

        if weight.size != target.size:
            raise ValueError(
                f"weights and targets must be have the same shape, {weight.size} != {target.size}"
            )

        loss = loss * weight[target]

    return nova.mean(loss)


def nll_loss(
    log_probs: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: Literal["none", "sum", "mean"] = "mean",
) -> Tensor:

    log_probs = ensure_tensor(log_probs)
    target = ensure_tensor(target)
    N = log_probs.size[0]

    loss = -log_probs[nova.arange(N, dtype=nova.long), target.to(nova.long)]

    if weight is not None:
        if weight.size != target.size:
            raise ValueError(
                f"weights and targets must be have the same shape, {weight.size} != {target.size}"
            )

        loss = loss * weight[target]

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"reduction expect ('sum','mean','none'), got '{reduction}'")


def cross_entropy(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None
) -> Tensor:

    logits = ensure_tensor(input)
    log_probs = log_softmax(logits)

    return nll_loss(log_probs, target, weight=weight)


def kl_div(
    log_probs: Tensor,
    target: Tensor,
    log_target: bool = False,
    reduction: Literal["none", "batchmean", "sum", "mean"] = "mean",
) -> Tensor:

    log_probs = ensure_tensor(log_probs)
    target = ensure_tensor(target)

    eps = 1e-12

    if log_target:
        loss = nova.exp(target) * (target - log_probs)
    else:
        probs_target = nova.clamp(target, eps, 1.0)
        loss = probs_target * (nova.log(probs_target) - log_probs)

    loss = loss.sum(dim=1)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return nova.sum(loss)
    elif reduction == "mean":
        return nova.mean(loss)
    elif reduction == "batchmean":
        batch_size = log_probs.size[0]
        return nova.sum(loss) / batch_size
    else:
        raise ValueError(
            f"reduction expect ('sum','mean','none', 'batchmean'), got '{reduction}'"
        )


# layer ops


def linear(
    input: Tensor, weight: Tensor | Parameter, bias: Optional[Tensor | Parameter] = None
) -> Tensor:
    input = ensure_tensor(input)
    output = input @ weight.T
    if bias is not None:
        output = output + bias

    return output


def flatten(input: Tensor) -> Tensor:
    N = input.size[0]

    return input.reshape(N, -1)


def _pair(input: int | tuple[int, int]):

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


def conv1d(
    input: Tensor,
    weight: Tensor | Parameter,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    *,
    bias: Optional[Parameter] = None,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
):
    input = ensure_tensor(input)

    if input.dim() != 3:
        raise ValueError(f"Conv1d expect 3D tensors, got {input.dim()}")

    K = kernel_size
    S = stride
    P = padding

    def _calculate_out_size(L: int) -> int:
        L_out = (L + 2 * P - K) // S + 1
        return L_out

    def _add_padding(input: Tensor) -> Tensor:

        pad_width = ((0, 0), (0, 0), (P, P))
        modes = ("zeros", "reflect", "replicate", "circular")

        if padding_mode in modes:
            mode = (
                "constant"
                if padding_mode == "zeros"
                else (
                    "reflect"
                    if padding_mode == "reflect"
                    else "edge" if padding_mode == "replicate" else "wrap"
                )
            )
        else:
            raise ValueError(f"padding_mode Only accept {modes} not '{padding_mode}'")

        return nova.pad(input, pad_width, mode=mode)

    def _im2col(input: Tensor, input_size: tuple[int, int, int]) -> tuple[Tensor, int]:

        N, C, L = input_size

        input_padded = _add_padding(input)

        L_out = _calculate_out_size(L)

        size = (N, C, L_out, K)
        sN, sC, sL = input_padded.strides
        strides = (sN, sC, sL * S, sL)

        window = nova.as_strided(
            input_padded, size=size, strides=strides
        )  # (N, C, L_out, K)

        col = window.permute(1, 3, 0, 2).reshape(C * K, -1)  # (C*K,N*L_out)

        return col, L_out

    out_channels = weight.size[0]

    input_size = input.size

    N = input_size[0]

    col, L_out = _im2col(input, input_size)

    w_col = weight.reshape(out_channels, -1)

    out = w_col @ col  # -> (C_out, N*L_out)

    if bias is not None:

        out = out + bias.reshape(out_channels, -1)

    out = out.reshape(out_channels, N, L_out)
    out = out.permute(1, 0, 2)

    return out


def conv2d(
    input: Tensor,
    weight: Tensor | Parameter,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    *,
    bias: Optional[Parameter] = None,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
):

    input = ensure_tensor(input)

    if input.dim() != 4:
        raise ValueError(f"Conv2d expect 4D tensors, got {input.dim()}")

    KH, KW = _pair(kernel_size)
    PH, PW = _pair(padding)
    SH, SW = _pair(stride)

    def _calculate_out_size(H: int, W: int) -> tuple[int, int]:
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1
        return H_out, W_out

    def _add_padding(input: Tensor) -> Tensor:

        pad_width = ((0, 0), (0, 0), (PH, PH), (PW, PW))
        modes = ("zeros", "reflect", "replicate", "circular")

        if padding_mode in modes:
            mode = (
                "constant"
                if padding_mode == "zeros"
                else (
                    "reflect"
                    if padding_mode == "reflect"
                    else "edge" if padding_mode == "replicate" else "wrap"
                )
            )
        else:
            raise ValueError(f"padding_mode Only accept {modes} not '{padding_mode}'")

        return nova.pad(input, pad_width, mode=mode)

    def _im2col(
        input: Tensor, input_size: tuple[int, int, int, int]
    ) -> tuple[Tensor, int, int]:
        N, C, H, W = input_size

        input_padded = _add_padding(input)

        H_out, W_out = _calculate_out_size(H, W)

        size = (N, C, H_out, W_out, KH, KW)

        sN, sC, sH, sW = input_padded.strides
        strides = (sN, sC, sH * SH, sW * SW, sH, sW)

        window = nova.as_strided(input_padded, size=size, strides=strides)

        col = window.permute(1, 4, 5, 0, 2, 3).reshape(C * KH * KW, -1)

        return col, H_out, W_out

    out_channels = weight.size[0]
    input_size = input.size

    N = input_size[0]

    w_col = weight.reshape(out_channels, -1)

    col, H_out, W_out = _im2col(input=input, input_size=input_size)

    out = w_col @ col  # -> c_out, N, H_out, W_out

    if bias is not None:
        out = out + bias.reshape(out_channels, -1)

    out = out.reshape(out_channels, N, H_out, W_out)
    out = out.permute(1, 0, 2, 3)  # output size -> (N, C_out, H_out, W_out)

    return out


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


def conv3d(
    input: Tensor,
    weight: Tensor | Parameter,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    *,
    bias: Optional[Parameter] = None,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
) -> Tensor:
    input = ensure_tensor(input)

    if input.dim() != 5:
        raise ValueError(f"Con3d expect 5D tensors, got {input.dim()}")

    KD, KH, KW = _triple(kernel_size)
    SD, SH, SW = _triple(stride)
    PD, PH, PW = _triple(padding)

    def _calculate_out_size(D: int, H: int, W: int) -> tuple[int, int, int]:
        D_out = (D + 2 * PD - KD) // SD + 1
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1
        return D_out, H_out, W_out

    def _add_padding(input: Tensor) -> Tensor:
        pad_width = ((0, 0), (0, 0), (PD, PD), (PH, PH), (PW, PW))
        modes = ("zeros", "reflect", "replicate", "circular")

        if padding_mode in modes:
            mode = (
                "constant"
                if padding_mode == "zeros"
                else (
                    "reflect"
                    if padding_mode == "reflect"
                    else "edge" if padding_mode == "replicate" else "wrap"
                )
            )
        else:
            raise ValueError(f"padding_mode Only accept {modes} not '{padding_mode}'")

        return nova.pad(input, pad_width, mode=mode)

    def _im2col(
        input: Tensor, input_size: tuple[int, int, int, int, int]
    ) -> tuple[Tensor, int, int, int]:
        N, C, D, H, W = input_size
        input_padded = _add_padding(input=input)
        D_out, H_out, W_out = _calculate_out_size(D, H, W)

        size = (N, C, D_out, H_out, W_out, KD, KH, KW)

        sN, sC, sD, sH, sW = input_padded.strides

        strides = (sN, sC, sD * SD, sH * SH, sW * SW, sD, sH, sW)

        window = nova.as_strided(input_padded, size=size, strides=strides)

        # window size -> (N, C, D_out, H_out, W_out, KD, KH, KW)
        col = window.permute(1, 5, 6, 7, 0, 2, 3, 4).reshape(C * KD * KH * KW, -1)

        return col, D_out, H_out, W_out

    out_channels = weight.size[0]
    input_size = input.size
    N = input_size[0]

    col, D_out, H_out, W_out = _im2col(input=input, input_size=input_size)

    w_col = weight.reshape(out_channels, -1)

    out = w_col @ col

    if bias is not None:
        out = out + bias.reshape(out_channels, -1)

    out = out.reshape(out_channels, N, D_out, H_out, W_out)
    out = out.permute(1, 0, 2, 3, 4)

    return out


def conv_transpose1d() -> Tensor: ...
def conv_transpose2d() -> Tensor: ...
def conv_transpose3d() -> Tensor: ...


def avg_pool1d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor:
    input = ensure_tensor(input)

    if input.dim() != 3:
        raise ValueError(f"AvgPool1d expect 3D tensors, got {input.dim()}")

    K = kernel_size
    S = stride if stride is not None else K
    P = padding

    def _calculate_out_size(L: int) -> int:
        L_out = (L + 2 * P - K) // S + 1
        return L_out

    def _add_padding(input: Tensor) -> Tensor:

        pad_width = ((0, 0), (0, 0), (P, P))
        return nova.pad(input, pad_width, mode="constant")

    N, C, L = input.size

    input_padded = _add_padding(input)

    L_out = _calculate_out_size(L)

    # Shape of the windows: (N, C, L_out, K)
    shape = (N, C, L_out, kernel_size)
    sN, sC, sL = input_padded.strides
    strides = (sN, sC, sL * S, sL)

    return nova.as_strided(input_padded, shape=shape, strides=strides).mean(dim=3)


def avg_pool2d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor:
    input = ensure_tensor(input)

    KH, KW = _pair(kernel_size)
    SH, SW = _pair(stride) if stride is not None else KH, KW
    PH, PW = _pair(padding)

    if input.dim() != 4:
        raise ValueError(f"AvgPool2d expect 4D tensors, got {input.dim()}")

    def _calculate_out_size(H: int, W: int) -> tuple[int, int]:
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1
        return H_out, W_out

    def _add_padding(input: Tensor) -> Tensor:

        pad_width = ((0, 0), (0, 0), (PH, PH), (PW, PW))

        return nova.pad(input, pad_width=pad_width, mode="constant")

    N, C, H, W = input.size

    input_padded = _add_padding(input)
    H_out, W_out = _calculate_out_size(H, W)

    size = (N, C, H_out, W_out, KH, KW)

    sN, sC, sH, sW = input_padded.strides
    strides = (sN, sC, sH * SH, sW * SW, sH, sW)

    return nova.as_strided(input_padded, size=size, strides=strides).mean(dim=(4, 5))


def avg_pool3d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor:

    input = ensure_tensor(input)

    if input.dim() != 5:
        raise ValueError(f"AvgPool3d expect 5D tensors, got {input.dim()}")

    KD, KH, KW = _triple(kernel_size)
    SD, SH, SW = _triple(stride) if stride is not None else KD, KH, KW
    PD, PH, PW = _triple(padding)

    def _calculate_out_size(D: int, H: int, W: int) -> tuple[int, int, int]:
        D_out = (D + 2 * PD - KD) // SD + 1
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1
        return D_out, H_out, W_out

    def _add_padding(input: Tensor) -> Tensor:
        pad_width = ((0, 0), (0, 0), (PD, PD), (PH, PH), (PW, PW))
        return nova.pad(input, pad_width, mode="constant")

    N, C, D, H, W = input.size
    input_padded = _add_padding(input=input)
    D_out, H_out, W_out = _calculate_out_size(D, H, W)

    size = (N, C, D_out, H_out, W_out, KD, KH, KW)

    sN, sC, sD, sH, sW = input_padded.strides

    strides = (sN, sC, sD * SD, sH * SH, sW * SW, sD, sH, sW)

    return nova.as_strided(input_padded, size=size, strides=strides).mean(dim=(5, 6, 7))


def adaptive_avg_pool1d(input: Tensor, output_size: int) -> Tensor:

    input = ensure_tensor(input)

    if input.dim() != 3:
        raise ValueError(f"AdaptativeAvgPool1d expect 1D tensors, got {input.dim()}")

    target_L = output_size
    L = input.size[2]

    if L == 1:
        return input.mean(dim=2)

    stride_L = L // target_L
    kernel_L = L - (target_L - 1) * stride_L

    return avg_pool1d(input, kernel_L, stride_L)


def adaptive_avg_pool2d(input: Tensor, output_size: tuple[int, int]) -> Tensor:

    input = ensure_tensor(input)

    if input.dim() != 4:
        raise ValueError(f"AdaptativeAvgPool2d expect 4D tensors, got {input.dim()}")

    H, W = input.size[2], input.size[3]
    target_H, target_W = _pair(output_size)

    if target_H == 1 and target_W == 1:
        return input.mean(dim=(2, 3))

    stride_H = H // target_H
    stride_W = W // target_W
    kernel_H = H - (target_H - 1) * stride_H
    kernel_W = W - (target_W - 1) * stride_W

    return avg_pool2d(
        input, kernel_size=(kernel_H, kernel_W), stride=(stride_H, stride_W)
    )


def adaptive_avg_pool3d(input: Tensor, output_size: tuple[int, int, int]) -> Tensor:

    input = ensure_tensor(input)

    if input.dim() != 5:
        raise ValueError(f"AdaptativeAvgPool3d expect 5D tensors, got {input.dim()}")

    D, H, W = input.size[2], input.size[3], input.size[4]
    target_D, target_H, target_W = _triple(output_size)

    if target_D == 1 and target_H == 1 and target_W == 1:
        return input.mean(dim=(2, 3, 4))

    stride_D = D // target_D
    stride_H = H // target_H
    stride_W = W // target_W
    kernel_D = D - (target_D - 1) * stride_D
    kernel_H = H - (target_H - 1) * stride_H
    kernel_W = W - (target_W - 1) * stride_W

    return avg_pool3d(
        input,
        kernel_size=(kernel_D, kernel_H, kernel_W),
        stride=(stride_D, stride_H, stride_W),
    )


def max_pool1d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor:
    input = ensure_tensor(input)

    if input.dim() != 3:
        raise ValueError(f"MaxPool1d expect 3D tensors, got {input.dim()}")

    K = kernel_size
    S = stride if stride is not None else K
    P = padding

    def _calculate_out_size(L: int) -> int:
        L_out = (L + 2 * P - K) // S + 1
        return L_out

    def _add_padding(input: Tensor) -> Tensor:

        pad_width = ((0, 0), (0, 0), (P, P))
        return nova.pad(input, pad_width, mode="constant")

    N, C, L = input.size

    input_padded = _add_padding(input)

    L_out = _calculate_out_size(L)

    # Shape of the windows: (N, C, L_out, K)
    shape = (N, C, L_out, kernel_size)
    sN, sC, sL = input_padded.strides
    strides = (sN, sC, sL * S, sL)

    return nova.as_strided(input_padded, shape=shape, strides=strides).max(dim=3)


def max_pool2d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor:
    input = ensure_tensor(input)

    KH, KW = _pair(kernel_size)
    PH, PW = _pair(padding)
    if stride is not None:
        SH, SW = _pair(stride)
    else:
        SH, SW = KH, KW

    if input.dim() != 4:
        raise ValueError(f"MaxPool2d expect 4D tensors, got {input.dim()}")

    def _calculate_out_size(H: int, W: int) -> tuple[int, int]:
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1
        return H_out, W_out

    def _add_padding(input: Tensor) -> Tensor:

        pad_width = ((0, 0), (0, 0), (PH, PH), (PW, PW))

        return nova.pad(input, pad_width=pad_width, mode="constant")

    N, C, H, W = input.size

    input_padded = _add_padding(input)
    H_out, W_out = _calculate_out_size(H, W)

    size = (N, C, H_out, W_out, KH, KW)

    sN, sC, sH, sW = input_padded.strides
    strides = (sN, sC, sH * SH, sW * SW, sH, sW)

    return nova.as_strided(input_padded, size=size, strides=strides).max(dim=(4, 5))


def max_pool3d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor:

    input = ensure_tensor(input)

    if input.dim() != 5:
        raise ValueError(f"MaxPool3d expect 5D tensors, got {input.dim()}")

    KD, KH, KW = _triple(kernel_size)
    PD, PH, PW = _triple(padding)
    if stride is not None:
        SD, SH, SW = _triple(stride)
    else:
        SD, SH, SW = KD, KH, KW

    def _calculate_out_size(D: int, H: int, W: int) -> tuple[int, int, int]:
        D_out = (D + 2 * PD - KD) // SD + 1
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1
        return D_out, H_out, W_out

    def _add_padding(input: Tensor) -> Tensor:
        pad_width = ((0, 0), (0, 0), (PD, PD), (PH, PH), (PW, PW))
        return nova.pad(input, pad_width, mode="constant")

    N, C, D, H, W = input.size
    input_padded = _add_padding(input=input)
    D_out, H_out, W_out = _calculate_out_size(D, H, W)

    size = (N, C, D_out, H_out, W_out, KD, KH, KW)

    sN, sC, sD, sH, sW = input_padded.strides

    strides = (sN, sC, sD * SD, sH * SH, sW * SW, sD, sH, sW)

    return nova.as_strided(input_padded, size=size, strides=strides).max(dim=(4, 5, 6))


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor | Buffer],
    running_var: Optional[Tensor | Buffer],
    weight: Optional[Tensor | Parameter] = None,
    bias: Optional[Tensor | Parameter] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> Tensor:

    input = ensure_tensor(input)

    if len(input.size) < 2:
        raise ValueError(f"expected at last 2D input, go {len(input.size)}")

    num_features = input.size[1]

    if training:
        batch_size = input.size[0]
        dims_to_reduce = [0] + list(range(2, input.dim()))

        mu = nova.mean(input, dim=dims_to_reduce, keepdims=True)
        var_biased = nova.var(input, dim=dims_to_reduce, keepdims=True)

        normalized = (input - mu) / nova.sqrt(var_biased + eps)

        if running_mean is not None and running_var is not None:
            var_unbiased = (
                var_biased * (batch_size / (batch_size - 1))
                if batch_size > 1
                else var_biased
            )

            current_mu = mu.reshape(-1)
            current_var = var_unbiased.reshape(-1)

            with nova.no_grad():
                running_mean.copy_(
                    (1 - momentum) * running_mean + momentum * current_mu
                )

                running_var.copy_((1 - momentum) * running_var + momentum * current_var)
    else:

        if running_mean is None or running_var is None:
            raise ValueError(
                "In evaluation mode, running_mean and running_var must be provided"
            )

        mean_shape = [1, num_features] + [1] * (input.dim() - 2)
        var_shape = mean_shape

        mean_broadcast = running_mean.reshape(*mean_shape)
        var_broadcast = running_var.reshape(*var_shape)

        normalized = (input - mean_broadcast) / nova.sqrt(var_broadcast + eps)

    if weight is not None:
        weight = ensure_tensor(weight)

        weight_shape = [1, num_features] + [1] * (input.dim() - 2)
        weight_broadcast = weight.reshape(*weight_shape)
        normalized = normalized * weight_broadcast

    if bias is not None:
        bias = ensure_tensor(bias)

        bias_shape = [1, num_features] + [1] * (input.dim() - 2)
        bias_broadcast = bias.reshape(*bias_shape)
        normalized = normalized + bias_broadcast

    return normalized


def layer_norm(
    input: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-05,
):

    input = ensure_tensor(input)

    input_shape = input.size
    if len(normalized_shape) > len(input_shape):
        raise ValueError(
            f"normalized_shape {normalized_shape} tiene más dimensiones "
            f"que el input shape {input_shape}"
        )

    for i, dim_size in enumerate(normalized_shape):
        input_dim = input_shape[-(len(normalized_shape) - i)]
        if dim_size != input_dim:
            raise ValueError(
                f"normalized_shape {normalized_shape} no coincide con "
                f"las últimas dimensiones de input {input_shape}"
            )

    num_dim_to_normalize = len(normalized_shape)
    dims_to_normalize = tuple(range(-num_dim_to_normalize, 0))
    mean = nova.mean(input, dim=dims_to_normalize, keepdims=True)

    variance = nova.mean((input - mean) ** 2, dim=dims_to_normalize, keepdims=True)

    normalized = (input - mean) / nova.sqrt(variance + eps)

    if weight is not None:
        weight = ensure_tensor(weight)

        num_leading_dims = len(input_shape) - len(normalized_shape)
        for _ in range(num_leading_dims):
            weight = weight.unsqueeze(0)

        normalized = normalized * weight

    if bias is not None:
        bias = ensure_tensor(bias)
        num_leading_dims = len(input_shape) - len(normalized_shape)
        for _ in range(num_leading_dims):
            bias = bias.unsqueeze(0)

        normalized = normalized + bias

    return normalized


def normalize(input: Tensor, p: int = 2, dim: Dim = 1, eps: float = 1e-12) -> Tensor:

    input = ensure_tensor(input)

    norm = nova.norm(input, ord=p, dim=dim, keepdims=True)

    return input / norm


def dropout(input: Tensor, p: float = 0.5, training: bool = True):
    if not training or p == 0:
        return input

    if p < 0 or p >= 1:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")

    input = ensure_tensor(input)

    mask_bool = nova.rand(*input.size) > p
    mask = ensure_tensor(mask_bool, dtype=input.dtype)
    mask = mask / (1 - p)
    return input * mask


def dropout2d(input: Tensor, p: float = 0.5, training: bool = True):

    if not training or p == 0:
        return input

    if p < 0 or p >= 1:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")

    input = ensure_tensor(input)

    if input.dim() != 4:
        raise ValueError(f"dropout2d expected input 4D, got{input.dim()}")

    N, C = input.size[0], input.size[1]

    mask_size = (N, C, 1, 1)

    mask_bool = nova.rand(*mask_size) > p
    mask = ensure_tensor(mask_bool, dtype=input.dtype)
    mask = mask / (1 - p)
    return input * mask


def dropout3d(input: Tensor, p: float = 0.5, training: bool = True):

    if not training or p == 0:
        return input

    if p < 0 or p >= 1:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")

    input = ensure_tensor(input)

    if input.dim() != 5:
        raise ValueError(f"dropout3d expected input 5D, got{input.dim()}")

    N, C = input.size[0], input.size[1]

    mask_size = (N, C, 1, 1, 1)

    mask_bool = nova.rand(*mask_size) > p
    mask = ensure_tensor(mask_bool, dtype=input.dtype)
    mask = mask / (1 - p)
    return input * mask
