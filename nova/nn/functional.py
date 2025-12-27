from __future__ import annotations

from pyparsing import Optional
import nova
from typing import TYPE_CHECKING, Literal
from nova.utils import ensure_tensor

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dim, KernelSize, Stride, Padding
    from nova.nn import Parameter, Buffer


# Activations


def relu(input: Tensor) -> Tensor: ...


def leaky_relu(input: Tensor, alpha: float = 0.01) -> Tensor: ...


def gelu(input: Tensor) -> Tensor: ...


def prelu(input: Tensor, alpha: float = 0.1) -> Tensor: ...


def sigmoid(input: Tensor) -> Tensor: ...


def tanh(input: Tensor) -> Tensor: ...


def softmax(input: Tensor, dim: Dim = 1) -> Tensor: ...


def log_softmax(input: Tensor, dim: Dim = 1) -> Tensor: ...


# criterion
def mse_loss(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None
) -> Tensor: ...


def binary_cross_entropy(input: Tensor, target: Tensor) -> Tensor: ...


def binary_cross_entropy_with_logits(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None
) -> Tensor: ...


def cross_entropy(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None
) -> Tensor: ...


def l1_loss(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None
) -> Tensor: ...


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    beta: float = 1.0,
    reduction: Literal["none", "mean", "sum"] = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor: ...


def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: Literal["none", "sum", "mean"] = "mean",
) -> Tensor: ...


def kl_div(
    input: Tensor,
    target: Tensor,
    reduction: Literal["none", "batchmean", "sum", "mean"] = "mean",
    log_target: bool = False,
    weight: Optional[Tensor] = None,
) -> Tensor: ...


# layer ops


def linear(
    input: Tensor, weight: Tensor | Parameter, bias: Optional[Tensor | Parameter] = None
) -> Tensor: ...


def flatten(input: Tensor) -> Tensor: ...


def conv1d(
    input: Tensor,
    weight: Parameter | Tensor,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    *,
    bias: Optional[Parameter | Tensor] = None,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
) -> Tensor: ...


def conv2d(
    input: Tensor,
    weight: Parameter | Tensor,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    *,
    bias: Optional[Parameter | Tensor] = None,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
) -> Tensor: ...


def conv3d(
    input: Tensor,
    weight: Parameter | Tensor,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    *,
    bias: Optional[Parameter | Tensor] = None,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
) -> Tensor: ...


def conv_transpose1d() -> Tensor: ...
def conv_transpose2d() -> Tensor: ...
def conv_transpose3d() -> Tensor: ...


def avg_pool1d() -> Tensor: ...
def avg_pool2d() -> Tensor: ...
def avg_pool3d() -> Tensor: ...


def max_pool1d() -> Tensor: ...
def max_pool2d() -> Tensor: ...
def max_pool3d() -> Tensor: ...


def adaptive_avg_pool1d() -> Tensor: ...
def adaptive_avg_pool2d() -> Tensor: ...
def adaptive_avg_pool3d() -> Tensor: ...


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor | Buffer],
    running_var: Optional[Tensor | Buffer],
    weight: Optional[Tensor | Parameter] = None,
    bias: Optional[Tensor | Parameter] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> Tensor: ...


def layer_norm(
    input: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Optional[Tensor | Parameter] = None,
    bias: Optional[Tensor | Parameter] = None,
    eps: float = 1e-05,
) -> Tensor: ...


def normalize(
    input: Tensor, p: float = 2.0, dim: Dim = 1, eps: float = 1e-12
) -> Tensor: ...


def dropout1d(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor: ...
def dropout2d(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor: ...
def dropout3d(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor: ...


# TODO: Add more methods
