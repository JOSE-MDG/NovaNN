from typing import Optional
from nova import Tensor
from nova.nn import Parameter, Buffer
from nova._typing import (
    Dim,
    LossReduction,
    KernelSize,
    Stride,
    Padding,
    PaddingMode,
    Dilation,
    Size,
)

# Activation Functions
def relu(input: Tensor) -> Tensor: ...
def leaky_relu(input: Tensor, alpha: float = 0.01) -> Tensor: ...
def gelu(input: Tensor) -> Tensor: ...
def prelu(input: Tensor, weight: float = 0.25) -> Tensor: ...
def sigmoid(input: Tensor) -> Tensor: ...
def tanh(input: Tensor) -> Tensor: ...
def softmax(input: Tensor, dim: Dim = 1) -> Tensor: ...
def log_softmax(input: Tensor, dim: Dim = 1) -> Tensor: ...

# Loss Functions
def _reduce(
    loss: Tensor,
    reduction_mode: LossReduction = "mean",
    batch_size: Optional[int] = None,
) -> Tensor: ...
def mse_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "mean",
) -> Tensor: ...
def l1_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "mean",
) -> Tensor: ...
def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    beta: float = 1.0,
    reduction: LossReduction = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor: ...
def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "mean",
) -> Tensor: ...
def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "mean",
    pos_weight: Optional[Tensor] = None,
) -> Tensor: ...
def nll_loss(
    log_probs: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "mean",
) -> Tensor: ...
def cross_entropy(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None
) -> Tensor: ...
def kl_div(
    log_probs: Tensor,
    target: Tensor,
    log_target: bool = False,
    reduction: LossReduction = "mean",
) -> Tensor: ...

# Linear and Convolutional Layers
def linear(
    input: Tensor, weight: Tensor | Parameter, bias: Optional[Tensor | Parameter] = None
) -> Tensor: ...
def flatten(input: Tensor, start_dim: int = 1, end_dim: int = -1) -> Tensor: ...
def conv1d(
    input: Tensor,
    weight: Tensor | Parameter,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    dilation: Dilation = 1,
    *,
    bias: Optional[Parameter] = None,
    padding_mode: PaddingMode = "zeros",
) -> Tensor: ...
def conv2d(
    input: Tensor,
    weight: Tensor | Parameter,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    dilation: Dilation = 1,
    *,
    bias: Optional[Parameter] = None,
    padding_mode: PaddingMode = "zeros",
) -> Tensor: ...
def conv3d(
    input: Tensor,
    weight: Tensor | Parameter,
    kernel_size: KernelSize,
    stride: Stride = 1,
    padding: Padding = 0,
    dilation: Dilation = 1,
    *,
    bias: Optional[Parameter] = None,
    padding_mode: PaddingMode = "zeros",
) -> Tensor: ...

# Pooling Layers
def avg_pool1d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor: ...
def avg_pool2d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor: ...
def avg_pool3d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor: ...
def global_avg_pool1d(input: Tensor) -> Tensor: ...
def global_avg_pool2d(input: Tensor) -> Tensor: ...
def global_avg_pool3d(input: Tensor) -> Tensor: ...
def max_pool1d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
    dilation: Dilation = 1,
) -> Tensor: ...
def max_pool2d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
    dilation: Dilation = 1,
) -> Tensor: ...
def max_pool3d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
    dilation: Dilation = 1,
) -> Tensor: ...

# Normalization
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
    normalized_shape: Size,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-05,
) -> Tensor: ...
def normalize(input: Tensor, p: int = 2, dim: Dim = 1) -> Tensor: ...

# Dropout
def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor: ...
def dropout2d(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor: ...
def dropout3d(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor: ...
