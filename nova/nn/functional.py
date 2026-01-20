from __future__ import annotations
import nova
from typing import TYPE_CHECKING, Optional
from nova.utils import ensure_tensor
from nova.autograd._ops import ReLU, LeakyReLU, PReLU, GELU, Sigmoid
from nova.nn.utils.standardization import _single, _pair, _triple

if TYPE_CHECKING:
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


def relu(input: Tensor) -> Tensor:
    """
    Applies the Rectified Linear Unit activation function element-wise.

    Forward: ReLU(x) = max(0, x)

    Args:
        input: Input tensor.

    Returns:
        Tensor with negative values clamped to zero.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> F.relu(x)
        tensor([0., 0., 0., 1., 2.], requires_grad=False, grad_fn=None, dtype=float32)
    """
    input = ensure_tensor(input)

    return ReLU.apply(input)


def leaky_relu(input: Tensor, alpha: float = 0.01) -> Tensor:
    """
    Applies the Leaky ReLU activation function element-wise.

    Forward: LeakyReLU(x) = max(alpha * x, x)

    Args:
        input: Input tensor.
        alpha: Negative slope coefficient. Controls the angle of the negative
            slope. Default is 0.01.

    Returns:
        Tensor with small negative slope for negative values.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> F.leaky_relu(x, alpha=0.1)
        tensor([-0.2, -0.1,  0. ,  1. ,  2. ], requires_grad=False, grad_fn=None, dtype=float32)
    """
    input = ensure_tensor(input)
    alpha = ensure_tensor(alpha)

    return LeakyReLU.apply(input, alpha)


def gelu(input: Tensor) -> Tensor:
    """
    Applies the Gaussian Error Linear Unit activation function.

    Forward: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    GELU is a smooth activation function that weights inputs by their value
    rather than gates inputs by their sign. Commonly used in transformers.

    Args:
        input: Input tensor.

    Returns:
        Tensor with GELU activation applied.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> F.gelu(x)
        tensor([-0.04540229, -0.158808  ,  0.        ,  0.841192  ,  1.9545977 ], requires_grad=False, grad_fn=None, dtype=float32)
    """
    input = ensure_tensor(input)
    return GELU.apply(input)


def prelu(input: Tensor, weight: float = 0.25) -> Tensor:
    """
    Applies the Parametric ReLU activation function element-wise.

    Forward: PReLU(x) = max(0, x) + weight * min(0, x)

    Args:
        input: Input tensor.
        weight: Learnable parameter controlling the negative slope.
            Default is 0.25.

    Returns:
        Tensor with parametric rectification applied.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> F.prelu(x, weight=0.25)
        tensor([-0.5 , -0.25,  0.  ,  1.  ,  2.  ], requires_grad=False, grad_fn=None, dtype=float32)
    """
    input = ensure_tensor(input)
    weight = ensure_tensor(weight)
    return PReLU.apply(input, weight)


def sigmoid(input: Tensor) -> Tensor:
    """
    Applies the Sigmoid activation function element-wise.

    Forward: σ(x) = 1 / (1 + e^(-x))

    Squashes values to the range (0, 1). Commonly used for binary
    classification and as gate activations in RNNs.

    Args:
        input: Input tensor.

    Returns:
        Tensor with values in range (0, 1).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> F.sigmoid(x)
        tensor([0.11920292, 0.26894143, 0.5       , 0.7310586 , 0.880797  ], requires_grad=False, grad_fn=None, dtype=float32)
    """
    input = ensure_tensor(input)
    return Sigmoid.apply(input)


def tanh(input: Tensor) -> Tensor:
    """
    Applies the hyperbolic tangent activation function element-wise.

    Forward: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Squashes values to the range (-1, 1). Zero-centered alternative to sigmoid.

    Args:
        input: Input tensor.

    Returns:
        Tensor with values in range (-1, 1).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> F.tanh(x)
        tensor([-0.9640276, -0.7615942,  0.       ,  0.7615942,  0.9640276], requires_grad=False, grad_fn=None, dtype=float32)
    """
    input = ensure_tensor(input)

    return input.tanh()


def softmax(input: Tensor, dim: Dim = 1) -> Tensor:
    """
    Applies the Softmax function along a dimension.

    Forward: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

    Converts logits to probabilities. Output sums to 1 along the specified
    dimension. Uses numerically stable implementation with max subtraction.

    Args:
        input: Input tensor (logits).
        dim: Dimension along which to compute softmax. Default is 1.

    Returns:
        Tensor with probabilities summing to 1 along dim.

    Examples:
        >>> import nova.nn.functional as F
        >>> logits = nova.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        >>> F.softmax(logits, dim=1)
        tensor([[0.09003057, 0.24472848, 0.66524094],
                [0.33333334, 0.33333334, 0.33333334]], requires_grad=False, grad_fn=None, dtype=float32)

    Notes:
        Uses the numerically stable formulation: softmax(x - max(x))
    """
    logits = ensure_tensor(input)

    # Numerical stability: subtract max to avoid overflow
    stable_logits = logits - nova.max(logits, dim=dim, keepdims=True)
    exp_logits = nova.exp(stable_logits)
    sum_exp = nova.sum(exp_logits, dim=dim, keepdims=True)
    out = exp_logits / sum_exp
    return out


def log_softmax(input: Tensor, dim: Dim = 1) -> Tensor:
    """
    Applies the log-softmax function along a dimension.

    Forward: log_softmax(x_i) = x_i - log(Σ_j exp(x_j))

    Computes log of softmax in a numerically stable way. Preferred over
    log(softmax(x)) for numerical stability and efficiency.

    Args:
        input: Input tensor (logits).
        dim: Dimension along which to compute log-softmax. Default is 1.

    Returns:
        Tensor with log-probabilities.

    Examples:
        >>> import nova.nn.functional as F
        >>> logits = nova.tensor([[1.0, 2.0, 3.0]])
        >>> F.log_softmax(logits, dim=1)
        tensor([[-2.407606  , -1.4076059 , -0.40760595]], requires_grad=False, grad_fn=None, dtype=float32)

    Notes:
        More numerically stable than torch.log(torch.softmax(x, dim)).
        Commonly used with NLLLoss for classification.
    """
    input = ensure_tensor(input)

    M = nova.max(input, dim=dim, keepdims=True)
    sum_exp = nova.sum(nova.exp(input - M), dim=dim, keepdims=True)
    out = (input - M) - nova.log(sum_exp)
    return out


# Loss Functions (Criterion)


def _reduce(
    loss: Tensor,
    reduction_mode: LossReduction = "mean",
    batch_size: Optional[int] = None,
) -> Tensor:
    """
    Applies reduction to loss tensor based on specified mode.

    Args:
        loss: Unreduced loss tensor.
        reduction_mode: Type of reduction to apply.
            - 'none': No reduction, returns full loss tensor
            - 'sum': Sums all elements
            - 'mean': Averages all elements
            - 'batchmean': Sum divided by batch size (for KL divergence)
        batch_size: Required only for 'batchmean' mode.

    Returns:
        Reduced loss tensor.

    Raises:
        ValueError: If reduction_mode is invalid or batch_size is missing
            for 'batchmean' mode.
    """
    if reduction_mode == "none":
        return loss
    elif reduction_mode == "sum":
        return nova.sum(loss)
    elif reduction_mode == "mean":
        return nova.mean(loss)
    elif reduction_mode == "batchmean":
        if batch_size is None:
            raise ValueError(
                "The batch size must be specified when the reduction is 'batchmean'"
            )
        return nova.sum(loss) / batch_size
    else:
        raise ValueError(
            f"reduction expect ('sum','mean','none', 'batchmean'), got '{reduction_mode}'"
        )


def mse_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "mean",
) -> Tensor:
    """
    Computes the Mean Squared Error loss.

    Forward: MSE = (target - input)²

    Measures the average squared difference between predictions and targets.
    Commonly used for regression tasks.

    Args:
        input: Predicted values.
        target: Ground truth values.
        weight: Optional element-wise weights. Must match target shape.
        reduction: Reduction mode ('none', 'mean', 'sum').

    Returns:
        Computed loss. Scalar if reduction != 'none', otherwise same shape as input.

    Raises:
        ValueError: If weight shape doesn't match target shape.

    Examples:
        >>> import nova.nn.functional as F
        >>> predictions = nova.tensor([2.5, 0.0, 2.0, 8.0])
        >>> targets = nova.tensor([3.0, -0.5, 2.0, 7.0])
        >>> F.mse_loss(predictions, targets)
        tensor(0.375, requires_grad=False, grad_fn=None, dtype=float32)
    """
    input = ensure_tensor(input)
    target = ensure_tensor(target)

    loss = (input - target) ** 2

    if weight is not None:
        weight = ensure_tensor(weight)

        if weight.shape != target.shape:
            raise ValueError(
                f"weights and targets must be have the same shape, {weight.shape} != {target.shape}"
            )

        loss = loss * weight

    return _reduce(loss, reduction)


def l1_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "mean",
) -> Tensor:
    """
    Computes the Mean Absolute Error (L1) loss.

    Forward: L1 = |target - input|

    Measures the average absolute difference between predictions and targets.
    More robust to outliers than MSE.

    Args:
        input: Predicted values.
        target: Ground truth values.
        weight: Optional element-wise weights. Must match target shape.
        reduction: Reduction mode ('none', 'mean', 'sum').

    Returns:
        Computed loss. Scalar if reduction != 'none', otherwise same shape as input.

    Examples:
        >>> import nova.nn.functional as F
        >>> predictions = nova.tensor([2.5, 0.0, 2.0, 8.0])
        >>> targets = nova.tensor([3.0, -0.5, 2.0, 7.0])
        >>> F.l1_loss(predictions, targets)
        tensor(0.5, requires_grad=False, grad_fn=None, dtype=float32)
    """
    logits = ensure_tensor(input)

    loss = nova.abs(logits - target)

    if weight is not None:
        weight = ensure_tensor(weight)

        if weight.shape != target.shape:
            raise ValueError(
                f"weights and targets must be have the same shape, {weight.shape} != {target.shape}"
            )

        loss = loss * weight

    return _reduce(loss, reduction)


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    beta: float = 1.0,
    reduction: LossReduction = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes the Smooth L1 loss (Huber loss).

    Forward:
        loss = 0.5 * (x²) / beta,  if |x| < beta
        loss = |x| - 0.5 * beta,   otherwise
    where x = target - input

    Combines benefits of L1 and L2 loss: less sensitive to outliers than MSE,
    but smoother than L1 near zero. Used in object detection (e.g., Faster R-CNN).

    Args:
        input: Predicted values.
        target: Ground truth values.
        beta: Threshold parameter. Smaller values make the loss more similar
            to L1, larger values to L2. Default is 1.0.
        weight: Optional element-wise weights. Must match target shape.
        reduction: Reduction mode ('none', 'mean', 'sum').

    Returns:
        Computed loss.

    Examples:
        >>> import nova.nn.functional as F
        >>> predictions = nova.tensor([2.5, 0.0, 2.0, 8.0])
        >>> targets = nova.tensor([3.0, -0.5, 2.0, 7.0])
        >>> F.smooth_l1_loss(predictions, targets, beta=1.0)
        tensor(0.1875, requires_grad=False, grad_fn=None, dtype=float32)
    """
    logits = ensure_tensor(input)
    target = ensure_tensor(target)
    beta = float(beta)

    diff = nova.abs(logits - target)
    condition = diff < beta
    loss = nova.where(condition, 0.5 * (diff**2) / beta, diff - 0.5 * beta)

    if weight is not None:
        weight = ensure_tensor(weight)

        if weight.shape != target.shape:
            raise ValueError(
                f"weights and targets must be have the same shape, {weight.shape} != {target.shape}"
            )

        loss = loss * weight

    return _reduce(loss, reduction)


def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "mean",
) -> Tensor:
    """
    Computes the Binary Cross Entropy loss.

    Forward: BCE = -(target * log(input) + (1 - target) * log(1 - input))

    Measures the performance of binary classification where inputs are
    probabilities (typically after sigmoid). For logits, use
    binary_cross_entropy_with_logits instead.

    Args:
        input: Predicted probabilities in [0, 1].
        target: Ground truth binary labels (0 or 1).
        weight: Optional element-wise weights. Must match target shape.
        reduction: Reduction mode ('none', 'mean', 'sum').

    Returns:
        Computed loss.

    Examples:
        >>> import nova.nn.functional as F
        >>> probs = nova.tensor([0.8, 0.3, 0.9, 0.2])
        >>> targets = nova.tensor([1.0, 0.0, 1.0, 0.0])
        >>> F.binary_cross_entropy(probs, targets)
        tensor(0.22708064, requires_grad=False, grad_fn=None, dtype=float32)

    Notes:
        Input values are clamped to [1e-12, 1-1e-12] for numerical stability.
    """
    input = ensure_tensor(input)
    target = ensure_tensor(target)

    eps = 1e-12

    loss = -(target * nova.log(input + eps) + (1 - target) * nova.log(1 - input + eps))

    if weight is not None:
        weight = ensure_tensor(weight)

        if weight.shape != target.shape:
            raise ValueError(
                f"weights and targets must be have the same shape, {weight.shape} != {target.shape}"
            )

        loss = loss * weight

    return _reduce(loss, reduction)


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "mean",
    pos_weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes Binary Cross Entropy loss with logits (numerically stable).

    This combines the sigmoid activation and the binary cross entropy loss
    in a single, stable operation to avoid overflow from large logits.

    Forward:
        loss = max(x, 0) - x * y + log(1 + exp(-|x|))

    With pos_weight:
        loss = max(x, 0) - x * y + (1 + (pos_weight - 1) * y) * log(1 + exp(-|x|))

    Args:
        input: Predicted logits (unbounded real values).
        target: Ground truth binary labels (0 or 1).
        weight: Optional element-wise weights. Must match target shape.
        reduction: Reduction mode ('none', 'mean', 'sum').
        pos_weight: Optional positive class weight for imbalanced datasets.

    Returns:
        The computed loss. Scalar if reduction != 'none', otherwise element-wise.

    Examples:
        >>> import nova.nn.functional as F
        >>> logits = nova.tensor([1.5, -0.5, 2.0, -1.0])
        >>> targets = nova.tensor([1.0, 0.0, 1.0, 0.0])
        >>> F.binary_cross_entropy_with_logits(logits, targets)
        tensor(0.27892, requires_grad=False, grad_fn=None, dtype=float32)
    """
    logits = ensure_tensor(input)
    target = ensure_tensor(target)

    # Numerically stable computation: max(x, 0) - x * y + log(1 + exp(-|x|))
    max_val = nova.maximum(logits, 0.0)

    if pos_weight is not None:
        pos_weight = ensure_tensor(pos_weight)
        # BCE with pos_weight:
        # loss = max(x,0) - x*y + (1 + (w-1)*y) * log(1 + exp(-|x|))
        log_term = nova.log(1.0 + nova.exp(-nova.abs(logits)))
        log_weight = 1.0 + (pos_weight - 1.0) * target
        loss = max_val - logits * target + log_weight * log_term
    else:
        # Standard BCE with logits (stable)
        loss = max_val - logits * target + nova.log(1.0 + nova.exp(-nova.abs(logits)))

    # Optional element-wise weighting
    if weight is not None:
        weight = ensure_tensor(weight)
        if weight.shape != target.shape:
            raise ValueError(
                f"weights and targets must have the same shape, got {weight.shape} != {target.shape}"
            )
        loss = loss * weight

    return _reduce(loss, reduction)


def nll_loss(
    log_probs: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: LossReduction = "mean",
) -> Tensor:
    """
    Computes the Negative Log Likelihood loss.

    Forward: NLL = -log_probs[target_class]

    Extracts the log probability of the correct class for each sample.
    Typically used after log_softmax. Combining log_softmax + nll_loss
    is equivalent to cross_entropy.

    Args:
        log_probs: Log probabilities from log_softmax, shape (N, C).
        target: Class indices (long tensor), shape (N,).
        weight: Optional per-class weights, shape (C,).
        reduction: Reduction mode ('none', 'mean', 'sum').

    Returns:
        Computed loss.

    Examples:
        >>> import nova.nn.functional as F
        >>> log_probs = F.log_softmax(nova.tensor([[2.0, 1.0, 0.1]]), dim=1)
        >>> target = nova.tensor([0], dtype=nova.long)
        >>> F.nll_loss(log_probs, target)
        tensor(0.41702995, requires_grad=False, grad_fn=None, dtype=float32)
    """
    log_probs = ensure_tensor(log_probs)
    target = ensure_tensor(target)
    N = log_probs.size(0)

    # Extract log probability of correct class for each sample
    loss = -log_probs[nova.arange(N), target]

    if weight is not None:
        if weight.shape[0] != log_probs.size(1):
            raise ValueError("weight must have same size as number of classes")

        loss = loss * weight[target]

    return _reduce(loss, reduction)


def cross_entropy(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None
) -> Tensor:
    """
    Computes the Cross Entropy loss for multi-class classification.

    Forward: CE = -log(softmax(input)[target_class])

    Combines log_softmax and nll_loss in one function. This is the standard
    loss for multi-class classification tasks.

    Args:
        input: Logits (pre-softmax), shape (N, C) where C is num classes.
        target: Class indices (long tensor), shape (N,).
        weight: Optional per-class weights, shape (C,).

    Returns:
        Mean cross entropy loss (scalar).

    Examples:
        >>> import nova.nn.functional as F
        >>> logits = nova.tensor([[2.0, 1.0, 0.1],
        ...                       [0.5, 2.5, 0.3]])
        >>> targets = nova.tensor([0, 1], dtype=nova.long)
        >>> F.cross_entropy(logits, targets)
        tensor(0.31853974, requires_grad=False, grad_fn=None, dtype=float32)

        >>> # With class weights (for imbalanced datasets)
        >>> class_weights = nova.tensor([0.5, 1.0, 2.0])
        >>> F.cross_entropy(logits, targets, weight=class_weights)
        tensor(0.21428224, requires_grad=False, grad_fn=None, dtype=float32)
    """
    logits = ensure_tensor(input)
    log_probs = log_softmax(logits)

    return nll_loss(log_probs, target, weight=weight)


def kl_div(
    log_probs: Tensor,
    target: Tensor,
    log_target: bool = False,
    reduction: LossReduction = "mean",
) -> Tensor:
    """
    Computes the Kullback-Leibler divergence loss.

    Forward:
        If log_target=False: KL = target * (log(target) - log_probs)
        If log_target=True:  KL = exp(target) * (target - log_probs)

    Measures how one probability distribution diverges from a second expected
    distribution. Useful for distillation, variational inference, and generative models.

    Args:
        log_probs: Log probabilities from model (typically after log_softmax).
        target: Target distribution. If log_target=False, these are probabilities.
            If log_target=True, these are log probabilities.
        log_target: If True, target is in log-space.
        reduction: Reduction mode ('none', 'mean', 'sum', 'batchmean').

    Returns:
        Computed KL divergence.

    Examples:
        >>> import nova.nn.functional as F
        >>> # Model outputs (student)
        >>> log_probs = F.log_softmax(nova.tensor([[2.0, 1.0, 0.5]]), dim=1)
        >>> # Teacher distribution
        >>> target_probs = nova.tensor([[0.7, 0.2, 0.1]])
        >>> F.kl_div(log_probs, target_probs, reduction='batchmean')
        tensor(0.01255022, requires_grad=False, grad_fn=None, dtype=float32)

    Notes:
        - KL divergence is not symmetric: KL(P||Q) ≠ KL(Q||P)
        - Always non-negative, zero when distributions are identical
        - 'batchmean' reduction divides by batch size (standard for KL)
    """
    log_probs = ensure_tensor(log_probs)
    target = ensure_tensor(target)

    eps = 1e-12

    if log_target:
        # Both inputs are in log space
        loss = nova.exp(target) * (target - log_probs)
    else:
        # Target is in probability space
        probs_target = nova.clamp(target, eps, 1.0)
        loss = probs_target * (nova.log(probs_target) - log_probs)

    loss = loss.sum(dim=1)
    batch_size = log_probs.size(0)
    return _reduce(loss, reduction, batch_size=batch_size)


# Linear and Convolutional Layers


def linear(
    input: Tensor, weight: Tensor | Parameter, bias: Optional[Tensor | Parameter] = None
) -> Tensor:
    """
    Applies a linear transformation to the incoming data: y = xWᵀ + b.

    This is the core operation of fully connected (dense) layers. It performs
    a matrix multiplication between the input and the transposed weight matrix,
    and optionally adds a bias term.

    Args:
        input: Input tensor of shape (N, in_features).
        weight: Weight tensor of shape (out_features, in_features).
        bias: Optional bias tensor of shape (out_features,). Default is None.

    Returns:
        Output tensor of shape (N, out_features).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(2, 3)
        >>> w = nova.randn(4, 3)
        >>> b = nova.randn(4)
        >>> F.linear(x, w, b).shape
        (2, 4)
    """
    input = ensure_tensor(input)
    output = input @ weight.T

    if bias is not None:
        out_features = weight.size(0)
        bias_view = bias.view(1, out_features)
        output = output + bias_view

    return output


def flatten(input: Tensor, start_dim: int = 1, end_dim: int = -1) -> Tensor:
    """
    Flattens a contiguous range of dimensions into a single dimension.

    Commonly used to prepare tensors for fully connected layers after
    convolutional layers.

    Args:
        input: Input tensor.
        start_dim: First dimension to flatten. Default is 1.
        end_dim: Last dimension to flatten. Default is -1.

    Returns:
        Flattened tensor with all specified dimensions merged into one.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(2, 3, 4, 4)
        >>> F.flatten(x).shape
        (2, 48)
    """
    input = ensure_tensor(input)
    return input.flatten(start_dim, end_dim)


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
) -> Tensor:
    """
    Applies a 1D convolution over an input signal composed of several input
    planes (channels).

    Forward:
        out(N, C_out, L_out) = W(C_out, C_in, K) * x(N, C_in, L_in) + bias

    Args:
        input: Input tensor of shape (N, C_in, L_in).
        weight: Filter tensor of shape (C_out, C_in, K).
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default is 1.
        padding: Padding added to both sides. Default is 0.
        dilation: Spacing between kernel elements. Default is 1.
        bias: Optional bias tensor of shape (C_out,). Default is None.
        padding_mode: Padding strategy ('zeros', 'reflect', 'replicate', 'circular').

    Returns:
        Output tensor of shape (N, C_out, L_out).

    Raises:
        ValueError: If input is not 3D or padding_mode is invalid.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 16)
        >>> w = nova.randn(6, 3, 3)
        >>> F.conv1d(x, w, kernel_size=3).shape
        (1, 6, 14)
    """
    input = ensure_tensor(input)

    if input.dim() != 3:
        raise ValueError(f"Conv1d expects 3D tensors, got {input.dim()}")

    K = _single(kernel_size)
    S = _single(stride)
    P = _single(padding)
    D = _single(dilation)

    def _calculate_out_size(L: int) -> int:
        K_eff = (K - 1) * D + 1
        L_out = (L + 2 * P - K_eff) // S + 1
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
            raise ValueError(f"padding_mode only accepts {modes}, not '{padding_mode}'")

        return nova.pad(input, pad_width, mode=mode)

    def _im2col(input: Tensor, input_size: tuple[int, int, int]) -> tuple[Tensor, int]:
        N, C, L = input_size
        input_padded = _add_padding(input)
        L_out = _calculate_out_size(L)

        size = (N, C, L_out, K)
        sN, sC, sL = input_padded.strides
        strides = (sN, sC, sL * S, sL * D)

        window = nova.as_strided(input_padded, size=size, strides=strides)
        col = window.permute(1, 3, 0, 2).reshape(C * K, -1)

        return col, L_out

    out_channels = weight.size(0)
    input_size = input.shape
    N = input_size[0]

    col, L_out = _im2col(input, input_size)
    w_col = weight.reshape(out_channels, -1)

    out = w_col @ col
    out = out.reshape(out_channels, N, L_out).permute(1, 0, 2)

    if bias is not None:
        bias_view = bias.view(1, out_channels, 1)
        out = out + bias_view

    return out


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
) -> Tensor:
    """
    Applies a 2D convolution over an input image composed of several input
    planes (channels).

    Forward:
        out(N, C_out, H_out, W_out) =
            W(C_out, C_in, KH, KW) * x(N, C_in, H_in, W_in) + bias

    Args:
        input: Input tensor of shape (N, C_in, H_in, W_in).
        weight: Filter tensor of shape (C_out, C_in, KH, KW).
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default is 1.
        padding: Padding added to all sides. Default is 0.
        dilation: Spacing between kernel elements. Default is 1.
        bias: Optional bias tensor of shape (C_out,). Default is None.
        padding_mode: Padding strategy ('zeros', 'reflect', 'replicate', 'circular').

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out).

    Raises:
        ValueError: If input is not 4D or padding_mode is invalid.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 32, 32)
        >>> w = nova.randn(8, 3, 3, 3)
        >>> F.conv2d(x, w, kernel_size=3).shape
        (1, 8, 30, 30)
    """
    input = ensure_tensor(input)

    if input.dim() != 4:
        raise ValueError(f"Conv2d expects 4D tensors, got {input.dim()}")

    KH, KW = _pair(kernel_size)
    PH, PW = _pair(padding)
    SH, SW = _pair(stride)
    DH, DW = _pair(dilation)

    def _calculate_out_size(H: int, W: int) -> tuple[int, int]:
        KH_eff = (KH - 1) * DH + 1
        KW_eff = (KW - 1) * DW + 1
        H_out = (H + 2 * PH - KH_eff) // SH + 1
        W_out = (W + 2 * PW - KW_eff) // SW + 1
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
            raise ValueError(f"padding_mode only accepts {modes}, not '{padding_mode}'")

        return nova.pad(input, pad_width, mode=mode)

    def _im2col(
        input: Tensor, input_size: tuple[int, int, int, int]
    ) -> tuple[Tensor, int, int]:
        N, C, H, W = input_size
        input_padded = _add_padding(input)
        H_out, W_out = _calculate_out_size(H, W)

        size = (N, C, H_out, W_out, KH, KW)
        sN, sC, sH, sW = input_padded.strides
        strides = (sN, sC, sH * SH, sW * SW, sH * DH, sW * DW)

        window = nova.as_strided(input_padded, size=size, strides=strides)
        col = window.permute(1, 4, 5, 0, 2, 3).reshape(C * KH * KW, -1)

        return col, H_out, W_out

    out_channels = weight.size(0)
    input_size = input.shape
    N = input_size[0]

    w_col = weight.reshape(out_channels, -1)
    col, H_out, W_out = _im2col(input, input_size)

    out = w_col @ col
    out = out.reshape(out_channels, N, H_out, W_out).permute(1, 0, 2, 3)

    if bias is not None:
        bias_view = bias.view(1, out_channels, 1, 1)
        out = out + bias_view

    return out


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
) -> Tensor:
    """
    Applies a 3D convolution over an input volume composed of several input
    planes (channels).

    Forward:
        out(N, C_out, D_out, H_out, W_out) =
            W(C_out, C_in, KD, KH, KW) * x(N, C_in, D_in, H_in, W_in) + bias

    This operation is used for volumetric data such as medical images (CT, MRI)
    or 3D feature maps in video models.

    Args:
        input: Input tensor of shape (N, C_in, D_in, H_in, W_in).
        weight: Filter tensor of shape (C_out, C_in, KD, KH, KW).
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default is 1.
        padding: Padding added to all spatial sides. Default is 0.
        dilation: Spacing between kernel elements. Default is 1.
        bias: Optional bias tensor of shape (C_out,). Default is None.
        padding_mode: Padding strategy ('zeros', 'reflect', 'replicate', 'circular').

    Returns:
        Output tensor of shape (N, C_out, D_out, H_out, W_out).

    Raises:
        ValueError: If input is not 5D or padding_mode is invalid.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 8, 16, 16)
        >>> w = nova.randn(6, 3, 3, 3, 3)
        >>> F.conv3d(x, w, kernel_size=3).shape
        (1, 6, 6, 14, 14)
    """
    input = ensure_tensor(input)

    if input.dim() != 5:
        raise ValueError(f"Conv3d expects 5D tensors, got {input.dim()}")

    KD, KH, KW = _triple(kernel_size)
    SD, SH, SW = _triple(stride)
    PD, PH, PW = _triple(padding)
    DD, DH, DW = _triple(dilation)

    def _calculate_out_size(D: int, H: int, W: int) -> tuple[int, int, int]:
        KD_eff = (KD - 1) * DD + 1
        KH_eff = (KH - 1) * DH + 1
        KW_eff = (KW - 1) * DW + 1
        D_out = (D + 2 * PD - KD_eff) // SD + 1
        H_out = (H + 2 * PH - KH_eff) // SH + 1
        W_out = (W + 2 * PW - KW_eff) // SW + 1
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
            raise ValueError(f"padding_mode only accepts {modes}, not '{padding_mode}'")

        return nova.pad(input, pad_width, mode=mode)

    def _im2col(
        input: Tensor, input_size: tuple[int, int, int, int, int]
    ) -> tuple[Tensor, int, int, int]:
        N, C, D, H, W = input_size
        input_padded = _add_padding(input=input)
        D_out, H_out, W_out = _calculate_out_size(D, H, W)

        size = (N, C, D_out, H_out, W_out, KD, KH, KW)
        sN, sC, sD, sH, sW = input_padded.strides
        strides = (sN, sC, sD * SD, sH * SH, sW * SW, sD * DD, sH * DH, sW * DW)

        window = nova.as_strided(input_padded, size=size, strides=strides)
        col = window.permute(1, 5, 6, 7, 0, 2, 3, 4).reshape(C * KD * KH * KW, -1)

        return col, D_out, H_out, W_out

    out_channels = weight.size(0)
    input_size = input.shape
    N = input_size[0]

    col, D_out, H_out, W_out = _im2col(input=input, input_size=input_size)
    w_col = weight.reshape(out_channels, -1)

    out = w_col @ col
    out = out.reshape(out_channels, N, D_out, H_out, W_out).permute(1, 0, 2, 3, 4)

    if bias is not None:
        bias_view = bias.view(1, out_channels, 1, 1, 1)
        out = out + bias_view

    return out


def avg_pool1d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor:
    """
    Applies a 1D average pooling over an input signal.

    Forward:
        out = average_pool1d(x)

    Reduces the input length by computing the mean over each sliding window.

    Args:
        input: Input tensor of shape (N, C, L_in).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling operation. Default is equal to kernel_size.
        padding: Implicit zero padding on both sides. Default is 0.

    Returns:
        Output tensor of shape (N, C, L_out).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 8)
        >>> F.avg_pool1d(x, kernel_size=2).shape
        (1, 3, 4)
    """
    input = ensure_tensor(input)

    if input.dim() != 3:
        raise ValueError(f"AvgPool1d expects 3D tensors, got {input.dim()}")

    K = _single(kernel_size)
    P = _single(padding)
    S = _single(stride) if stride is not None else K

    def _calculate_out_size(L: int) -> int:
        return (L + 2 * P - K) // S + 1

    def _add_padding(input: Tensor) -> Tensor:
        return nova.pad(input, ((0, 0), (0, 0), (P, P)), mode="constant")

    N, C, L = input.shape
    input_padded = _add_padding(input)
    L_out = _calculate_out_size(L)

    size = (N, C, L_out, K)
    sN, sC, sL = input_padded.strides
    strides = (sN, sC, sL * S, sL)

    return nova.as_strided(input_padded, size=size, strides=strides).mean(dim=3)


def avg_pool2d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
) -> Tensor:
    """
    Applies a 2D average pooling over an input image.

    Forward:
        out = average_pool2d(x)

    Reduces spatial dimensions by computing the mean of each window.

    Args:
        input: Input tensor of shape (N, C, H, W).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling operation. Default is equal to kernel_size.
        padding: Implicit zero padding on all sides. Default is 0.

    Returns:
        Output tensor of shape (N, C, H_out, W_out).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 8, 8)
        >>> F.avg_pool2d(x, kernel_size=2).shape
        (1, 3, 4, 4)
    """
    input = ensure_tensor(input)
    KH, KW = _pair(kernel_size)
    PH, PW = _pair(padding)
    SH, SW = _pair(stride) if stride is not None else (KH, KW)

    if input.dim() != 4:
        raise ValueError(f"AvgPool2d expects 4D tensors, got {input.dim()}")

    def _calculate_out_size(H: int, W: int) -> tuple[int, int]:
        return (H + 2 * PH - KH) // SH + 1, (W + 2 * PW - KW) // SW + 1

    def _add_padding(input: Tensor) -> Tensor:
        return nova.pad(input, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode="constant")

    N, C, H, W = input.shape
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
    """
    Applies a 3D average pooling over an input volume.

    Forward:
        out = average_pool3d(x)

    Reduces the spatial and depth dimensions by averaging each pooling region.

    Args:
        input: Input tensor of shape (N, C, D, H, W).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling operation. Default is equal to kernel_size.
        padding: Implicit zero padding on all sides. Default is 0.

    Returns:
        Output tensor of shape (N, C, D_out, H_out, W_out).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 8, 8, 8)
        >>> F.avg_pool3d(x, kernel_size=2).shape
        (1, 3, 4, 4, 4)
    """
    input = ensure_tensor(input)

    if input.dim() != 5:
        raise ValueError(f"AvgPool3d expects 5D tensors, got {input.dim()}")

    KD, KH, KW = _triple(kernel_size)
    PD, PH, PW = _triple(padding)
    SD, SH, SW = _triple(stride) if stride is not None else (KD, KH, KW)

    def _calculate_out_size(D: int, H: int, W: int) -> tuple[int, int, int]:
        return (
            (D + 2 * PD - KD) // SD + 1,
            (H + 2 * PH - KH) // SH + 1,
            (W + 2 * PW - KW) // SW + 1,
        )

    def _add_padding(input: Tensor) -> Tensor:
        return nova.pad(
            input, ((0, 0), (0, 0), (PD, PD), (PH, PH), (PW, PW)), mode="constant"
        )

    N, C, D, H, W = input.shape
    input_padded = _add_padding(input)
    D_out, H_out, W_out = _calculate_out_size(D, H, W)

    size = (N, C, D_out, H_out, W_out, KD, KH, KW)
    sN, sC, sD, sH, sW = input_padded.strides
    strides = (sN, sC, sD * SD, sH * SH, sW * SW, sD, sH, sW)

    return nova.as_strided(input_padded, size=size, strides=strides).mean(dim=(5, 6, 7))


def global_avg_pool1d(input: Tensor) -> Tensor:
    """
    Computes global average pooling over the temporal dimension.

    Collapses the entire sequence length into a single averaged value
    per channel.

    Args:
        input: Input tensor of shape (N, C, L).

    Returns:
        Output tensor of shape (N, C, 1).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 8)
        >>> F.global_avg_pool1d(x).shape
        (1, 3, 1)
    """
    return input.mean(dim=2, keepdims=True)


def global_avg_pool2d(input: Tensor) -> Tensor:
    """
    Computes global average pooling over spatial dimensions (H, W).

    Reduces each feature map to a single averaged value per channel.
    Often used before fully connected layers in classification models.

    Args:
        input: Input tensor of shape (N, C, H, W).

    Returns:
        Output tensor of shape (N, C, 1, 1).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 8, 8)
        >>> F.global_avg_pool2d(x).shape
        (1, 3, 1, 1)
    """
    return input.mean(dim=(2, 3), keepdims=True)


def global_avg_pool3d(input: Tensor) -> Tensor:
    """
    Computes global average pooling over volumetric dimensions (D, H, W).

    Aggregates the entire 3D feature map into a single value per channel.

    Args:
        input: Input tensor of shape (N, C, D, H, W).

    Returns:
        Output tensor of shape (N, C, 1, 1, 1).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 8, 8, 8)
        >>> F.global_avg_pool3d(x).shape
        (1, 3, 1, 1, 1)
    """
    return input.mean(dim=(2, 3, 4), keepdims=True)


def max_pool1d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
    dilation: Dilation = 1,
) -> Tensor:
    """
    Applies a 1D max pooling over an input signal.

    Forward:
        out = max_pool1d(x)

    Reduces the temporal dimension by selecting the maximum value in each
    pooling window.

    Args:
        input: Input tensor of shape (N, C, L_in).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling operation. Default is equal to kernel_size.
        padding: Implicit zero padding on both sides. Default is 0.
        dilation: Spacing between elements in the pooling window. Default is 1.

    Returns:
        Output tensor of shape (N, C, L_out).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 8)
        >>> F.max_pool1d(x, kernel_size=2).shape
        (1, 3, 4)
    """
    input = ensure_tensor(input)

    if input.dim() != 3:
        raise ValueError(f"MaxPool1d expects 3D tensors, got {input.dim()}")

    K = _single(kernel_size)
    P = _single(padding)
    D = _single(dilation)
    S = _single(stride) if stride is not None else K

    def _calculate_out_size(L: int) -> int:
        K_eff = (K - 1) * D + 1
        return (L + 2 * P - K_eff) // S + 1

    def _add_padding(input: Tensor) -> Tensor:
        return nova.pad(input, ((0, 0), (0, 0), (P, P)), mode="constant")

    N, C, L = input.shape
    input_padded = _add_padding(input)
    L_out = _calculate_out_size(L)

    shape = (N, C, L_out, K)
    sN, sC, sL = input_padded.strides
    strides = (sN, sC, sL * S, sL * D)

    return nova.as_strided(input_padded, size=shape, strides=strides).max(dim=3)


def max_pool2d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
    dilation: Dilation = 1,
) -> Tensor:
    """
    Applies a 2D max pooling over an input image.

    Forward:
        out = max_pool2d(x)

    Reduces spatial dimensions by selecting the maximum value in each
    pooling region.

    Args:
        input: Input tensor of shape (N, C, H_in, W_in).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling operation. Default is equal to kernel_size.
        padding: Implicit zero padding on all sides. Default is 0.
        dilation: Spacing between elements in the pooling window. Default is 1.

    Returns:
        Output tensor of shape (N, C, H_out, W_out).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 8, 8)
        >>> F.max_pool2d(x, kernel_size=2).shape
        (1, 3, 4, 4)
    """
    input = ensure_tensor(input)

    KH, KW = _pair(kernel_size)
    PH, PW = _pair(padding)
    DH, DW = _pair(dilation)
    SH, SW = _pair(stride) if stride is not None else (KH, KW)

    if input.dim() != 4:
        raise ValueError(f"MaxPool2d expects 4D tensors, got {input.dim()}")

    def _calculate_out_size(H: int, W: int) -> tuple[int, int]:
        KH_eff = (KH - 1) * DH + 1
        KW_eff = (KW - 1) * DW + 1
        return (H + 2 * PH - KH_eff) // SH + 1, (W + 2 * PW - KW_eff) // SW + 1

    def _add_padding(input: Tensor) -> Tensor:
        return nova.pad(input, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode="constant")

    N, C, H, W = input.shape
    input_padded = _add_padding(input)
    H_out, W_out = _calculate_out_size(H, W)

    size = (N, C, H_out, W_out, KH, KW)
    sN, sC, sH, sW = input_padded.strides
    strides = (sN, sC, sH * SH, sW * SW, sH * DH, sW * DW)

    return nova.as_strided(input_padded, size=size, strides=strides).max(dim=(4, 5))


def max_pool3d(
    input: Tensor,
    kernel_size: KernelSize,
    stride: Optional[Stride] = None,
    padding: Padding = 0,
    dilation: Dilation = 1,
) -> Tensor:
    """
    Applies a 3D max pooling over an input volume.

    Forward:
        out = max_pool3d(x)

    Reduces volumetric dimensions by taking the maximum value from each pooling
    region. Commonly used in 3D CNNs and video models.

    Args:
        input: Input tensor of shape (N, C, D_in, H_in, W_in).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling operation. Default is equal to kernel_size.
        padding: Implicit zero padding on all sides. Default is 0.
        dilation: Spacing between elements in the pooling window. Default is 1.

    Returns:
        Output tensor of shape (N, C, D_out, H_out, W_out).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(1, 3, 8, 8, 8)
        >>> F.max_pool3d(x, kernel_size=2).shape
        (1, 3, 4, 4, 4)
    """
    input = ensure_tensor(input)

    if input.dim() != 5:
        raise ValueError(f"MaxPool3d expects 5D tensors, got {input.dim()}")

    KD, KH, KW = _triple(kernel_size)
    PD, PH, PW = _triple(padding)
    DD, DH, DW = _triple(dilation)
    SD, SH, SW = _triple(stride) if stride is not None else (KD, KH, KW)

    def _calculate_out_size(D: int, H: int, W: int) -> tuple[int, int, int]:
        KD_eff = (KD - 1) * DD + 1
        KH_eff = (KH - 1) * DH + 1
        KW_eff = (KW - 1) * DW + 1
        return (
            (D + 2 * PD - KD_eff) // SD + 1,
            (H + 2 * PH - KH_eff) // SH + 1,
            (W + 2 * PW - KW_eff) // SW + 1,
        )

    def _add_padding(input: Tensor) -> Tensor:
        return nova.pad(
            input, ((0, 0), (0, 0), (PD, PD), (PH, PH), (PW, PW)), mode="constant"
        )

    N, C, D, H, W = input.shape
    input_padded = _add_padding(input)
    D_out, H_out, W_out = _calculate_out_size(D, H, W)

    size = (N, C, D_out, H_out, W_out, KD, KH, KW)
    sN, sC, sD, sH, sW = input_padded.strides
    strides = (sN, sC, sD * SD, sH * SH, sW * SW, sD * DD, sH * DH, sW * DW)

    return nova.as_strided(input_padded, size=size, strides=strides).max(dim=(5, 6, 7))


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
    """
    Applies Batch Normalization over a mini-batch of inputs.

    Normalizes input features by mean and variance, stabilizing and accelerating
    training. During training, statistics are computed per batch; during evaluation,
    running averages are used instead.

    Forward:
        y = (x - μ) / sqrt(σ² + eps) * weight + bias

    Args:
        input: Input tensor of shape (N, C, *).
        running_mean: Buffer tensor to store running mean (shape: (C,)).
        running_var: Buffer tensor to store running variance (shape: (C,)).
        weight: Optional learnable scale parameter (C,). Default is None.
        bias: Optional learnable shift parameter (C,). Default is None.
        training: Whether the layer is in training mode. Default is False.
        momentum: Momentum factor for updating running statistics. Default is 0.1.
        eps: Small constant for numerical stability. Default is 1e-5.

    Returns:
        Normalized tensor with same shape as input.

    Raises:
        ValueError: If input has fewer than 2 dimensions or running stats
        are missing during evaluation.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(4, 3, 8, 8)
        >>> running_mean = nova.zeros(3)
        >>> running_var = nova.ones(3)
        >>> F.batch_norm(x, running_mean, running_var, training=True).shape
        (4, 3, 8, 8)
    """
    input = ensure_tensor(input)

    if len(input.shape) < 2:
        raise ValueError(f"Expected at least 2D input, got {input.dim()}")

    num_features = input.size(1)

    if training:
        dims_to_reduce = [0] + list(range(2, input.dim()))
        mu = nova.mean(input, dim=dims_to_reduce, keepdims=True)
        var_biased = nova.var(input, dim=dims_to_reduce, keepdims=True)

        num_reduced = 1
        for d in dims_to_reduce:
            num_reduced *= input.size(d)

        var_unbiased = (
            var_biased * (num_reduced / (num_reduced - 1))
            if num_reduced > 1
            else var_biased
        )
        normalized = (input - mu) / nova.sqrt(var_biased + eps)

        if running_mean is not None and running_var is not None:
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
                "In evaluation mode, running_mean and running_var must be provided."
            )

        mean_shape = [1, num_features] + [1] * (input.dim() - 2)
        var_shape = mean_shape

        mean_broadcast = running_mean.reshape(*mean_shape)
        var_broadcast = running_var.reshape(*var_shape)
        normalized = (input - mean_broadcast) / nova.sqrt(var_broadcast + eps)

    if weight is not None:
        weight_shape = [1, num_features] + [1] * (input.dim() - 2)
        normalized = normalized * ensure_tensor(weight).reshape(*weight_shape)

    if bias is not None:
        bias_shape = [1, num_features] + [1] * (input.dim() - 2)
        normalized = normalized + ensure_tensor(bias).reshape(*bias_shape)

    return normalized


def layer_norm(
    input: Tensor,
    normalized_shape: Size,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-05,
) -> Tensor:
    """
    Applies Layer Normalization over the last certain number of dimensions.

    Normalizes across the specified dimensions of the input instead of across
    the batch, making it independent of batch size. Commonly used in
    Transformer architectures.

    Forward:
        y = (x - mean) / sqrt(var + eps) * weight + bias

    Args:
        input: Input tensor of arbitrary shape.
        normalized_shape: Shape of the dimensions to normalize (e.g., (C, H, W)).
        weight: Optional learnable scale parameter matching normalized_shape.
        bias: Optional learnable shift parameter matching normalized_shape.
        eps: Small constant added to variance for numerical stability. Default is 1e-5.

    Returns:
        Normalized tensor with same shape as input.

    Raises:
        ValueError: If normalized_shape is incompatible with input shape.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(2, 4, 8)
        >>> F.layer_norm(x, normalized_shape=(8,)).shape
        (2, 4, 8)
    """
    input = ensure_tensor(input)
    dtype = input.dtype
    input_shape = input.shape

    if not isinstance(normalized_shape, tuple):
        normalized_shape = (normalized_shape,)

    if len(normalized_shape) > len(input_shape):
        raise ValueError(
            f"normalized_shape {normalized_shape} has more dimensions than input {input_shape}"
        )

    for i, dim_size in enumerate(normalized_shape):
        input_dim = input_shape[-(len(normalized_shape) - i)]
        if dim_size != input_dim:
            raise ValueError(
                f"normalized_shape {normalized_shape} does not match the last dimensions of input {input_shape}"
            )

    dims_to_normalize = tuple(range(-len(normalized_shape), 0))
    mean = nova.mean(input, dim=dims_to_normalize, keepdims=True)
    variance = nova.mean((input - mean) ** 2, dim=dims_to_normalize, keepdims=True)

    eps = nova.tensor(eps, dtype=dtype)
    normalized = (input - mean) / nova.sqrt(variance + eps)

    if weight is not None:
        weight = ensure_tensor(weight)
        for _ in range(len(input_shape) - len(normalized_shape)):
            weight = weight.unsqueeze(0)
        normalized = normalized * weight

    if bias is not None:
        bias = ensure_tensor(bias)
        for _ in range(len(input_shape) - len(normalized_shape)):
            bias = bias.unsqueeze(0)
        normalized = normalized + bias

    return normalized


def normalize(input: Tensor, p: int = 2, dim: Dim = 1) -> Tensor:
    """
    Normalizes the tensor along a dimension using the p-norm.

    Forward:
        y = x / ||x||ₚ

    Commonly used for feature normalization or cosine similarity operations.

    Args:
        input: Input tensor.
        p: Order of the norm (e.g., 1 for L1, 2 for L2). Default is 2.
        dim: Dimension along which to compute the norm. Default is 1.

    Returns:
        Tensor with unit norm along the specified dimension.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.randn(2, 3)
        >>> F.normalize(x).shape
        (2, 3)
    """
    input = ensure_tensor(input)
    norm = nova.norm(input, ord=p, dim=dim, keepdims=True)
    return input / norm


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    Applies Dropout to the input tensor.

    Randomly zeroes some elements of the input tensor with probability `p`
    during training to prevent overfitting. Scales remaining values by 1/(1-p)
    to maintain the expected sum.

    Args:
        input: Input tensor.
        p: Probability of an element to be zeroed. Must be in [0, 1).
        training: Apply dropout only when True. Default is True.

    Returns:
        Tensor with randomly zeroed elements during training.

    Raises:
        ValueError: If p is outside [0, 1).

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.ones((2, 4))
        >>> F.dropout(x, p=0.5, training=True).shape
        (2, 4)
    """
    if not training or p == 0:
        return input

    if p < 0 or p >= 1:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")

    input = ensure_tensor(input)
    mask_bool = nova.rand(*input.shape) > p
    mask = ensure_tensor(mask_bool, dtype=input.dtype) / (1 - p)
    return input * mask


def dropout2d(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    Applies 2D Dropout to feature maps.

    Randomly zeroes entire channels (feature maps) of the input tensor with
    probability `p` during training, which helps promote independence between
    channels.

    Args:
        input: Input tensor of shape (N, C, H, W).
        p: Probability of a channel to be zeroed. Must be in [0, 1).
        training: Apply dropout only when True. Default is True.

    Returns:
        Tensor with dropped channels during training.

    Raises:
        ValueError: If input is not 4D or if p is invalid.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.ones((2, 3, 8, 8))
        >>> F.dropout2d(x, p=0.5, training=True).shape
        (2, 3, 8, 8)
    """
    if not training or p == 0:
        return input

    if p < 0 or p >= 1:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")

    input = ensure_tensor(input)

    if input.dim() != 4:
        raise ValueError(f"dropout2d expected 4D input, got {input.dim()}")

    N, C = input.size(0), input.size(1)
    mask_size = (N, C, 1, 1)

    mask_bool = nova.rand(*mask_size) > p
    mask = ensure_tensor(mask_bool, dtype=input.dtype) / (1 - p)
    return input * mask


def dropout3d(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    Applies 3D Dropout to volumetric feature maps.

    Randomly zeroes entire channels of the input tensor with probability `p`
    during training. This is the 3D analog of dropout2d, often used in 3D CNNs.

    Args:
        input: Input tensor of shape (N, C, D, H, W).
        p: Probability of a channel to be zeroed. Must be in [0, 1).
        training: Apply dropout only when True. Default is True.

    Returns:
        Tensor with dropped volumetric channels during training.

    Raises:
        ValueError: If input is not 5D or if p is invalid.

    Examples:
        >>> import nova.nn.functional as F
        >>> x = nova.ones((2, 3, 4, 4, 4))
        >>> F.dropout3d(x, p=0.3, training=True).shape
        (2, 3, 4, 4, 4)
    """
    if not training or p == 0:
        return input

    if p < 0 or p >= 1:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")

    input = ensure_tensor(input)

    if input.dim() != 5:
        raise ValueError(f"dropout3d expected 5D input, got {input.dim()}")

    N, C = input.size(0), input.size(1)
    mask_size = (N, C, 1, 1, 1)

    mask_bool = nova.rand(*mask_size) > p
    mask = ensure_tensor(mask_bool, dtype=input.dtype) / (1 - p)
    return input * mask
