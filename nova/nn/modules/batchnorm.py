from __future__ import annotations
import nova
import nova.nn.init as init
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module, LazyModuleMixin
from nova.nn.parameter import (
    Parameter,
    Buffer,
    UninitializedParameter,
    UninitializedBuffer,
)

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dtype


"""
Batch Normalization Behavior Table:

| training | track_running_stats | Result | Layer Behavior                                         |
|:---------|:--------------------|:-------|:-------------------------------------------------------|
| True     | True                | True   | Trains and updates running statistics (memory).        |
| True     | False               | True   | Trains using only the current batch (no memory).       |
| False    | False               | True   | Evaluates using current batch (because no memory).     |
| False    | True                | False  | Evaluates using saved memory (Standard production mode).|
"""

"""
Batch Normalization Usage Cheat Sheet:

| Parameter                 | When to change it           | Technical Reason                                         |
|:--------------------------|:----------------------------|:---------------------------------------------------------|
| momentum low (0.01)       | Noisy dataset or Fine-tuning| You want "memory" to be very stable and change slowly.   |
| momentum high (0.5)       | Clean/homogeneous dataset   | You want the model to adapt very quickly to the data.    |
| momentum = None           | Very small batches          | Uses num_batches_tracked to average history equally.     |
| track_running_stats=False | Siamese Nets / Meta-learning| Identical normalization in train/eval (input-based only).|
"""


class _BatchNorm(Module):
    """Base class for Batch Normalization layers.

    This is an internal base class that implements the core batch normalization
    logic. It should not be used directly - use BatchNorm1d, BatchNorm2d, or
    BatchNorm3d instead.

    Batch Normalization normalizes the input by subtracting the batch mean and
    dividing by the batch standard deviation. It then applies a learnable affine
    transformation.

    .. math::
        y = \\frac{x - \\text{E}[x]}{\\sqrt{\\text{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard deviation are calculated per-dimension over the mini-batches
    and :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors of size C
    (where C is the number of features or channels).

    During training, this layer keeps running estimates of its computed mean and
    variance, which are then used for normalization during evaluation.

    Args:
        num_features: Number of features or channels :math:`C` of the input
        momentum: Value used for running mean and variance computation. Can be set to
            ``None`` for cumulative moving average (i.e. simple average). Default: 0.1
        eps: Value added to the denominator for numerical stability. Default: 1e-5
        affine: If ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, this module tracks the running mean and variance,
            and uses them during evaluation instead of using batch statistics. Default: ``True``
        dtype: The desired data type of parameters and buffers. Default: None

    Attributes:
        weight: The learnable weights :math:`\\gamma` of shape (num_features,). Only created
            when ``affine=True``
        bias: The learnable bias :math:`\\beta` of shape (num_features,). Only created
            when ``affine=True``
        running_mean: The running mean of shape (num_features,). Only created when
            ``track_running_stats=True``
        running_var: The running variance of shape (num_features,). Only created when
            ``track_running_stats=True``
        num_batches_tracked: The number of batches tracked. Only created when
            ``track_running_stats=True``
    """

    running_mean: Buffer
    running_var: Buffer
    num_batches_tracked: Buffer

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        num_features: int,
        momentum: Optional[float] = 0.1,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()

        self.track_running_stats = track_running_stats
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.eps = eps

        if self.track_running_stats:
            self.running_mean = Buffer(nova.empty((num_features,), dtype=dtype))
            self.running_var = Buffer(nova.empty((num_features,)), dtype=dtype)
            self.num_batches_tracked = Buffer(nova.empty(()), dtype=nova.long)
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        if self.affine:
            self.weight = Parameter(nova.empty((num_features,)), dtype=dtype)
            self.bias = Parameter(nova.empty((num_features,)), dtype=dtype)
            self.weight.is_bn_param = True
            self.bias.is_bn_param = True
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets running statistics and learnable parameters.

        Running mean is set to 0, running variance to 1, and batch counter to 0.
        If affine parameters exist, weight is initialized to 1 and bias to 0.
        """
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.ones_()
            self.num_batches_tracked.zero_()

        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """Applies Batch Normalization to the input.

        Args:
            input: Input tensor

        Returns:
            Normalized tensor with same shape as input
        """
        self._check_input_dim(input)

        exp_avg_factor = 0.0

        if self._training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exp_avg_factor = 1.0 / float(self.num_batches_tracked.item())
                else:
                    exp_avg_factor = self.momentum

        return F.batch_norm(
            input=input,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=self._training or not self.track_running_stats,
            momentum=exp_avg_factor,
            eps=self.eps,
        )

    def _check_input_dim(self, input: Tensor) -> None | Exception:
        """Validates input tensor dimensions.

        Must be implemented by subclasses to check for appropriate dimensions.

        Args:
            input: Input tensor to validate

        Raises:
            NotImplementedError: Always raised as this must be implemented by subclasses
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        return "{num_features}, momentum={momentum}, eps={eps}, affine={affine}, track_running_stats={track_running_stats}".format(
            **self.__dict__
        )


class _LayzyNormBase(LazyModuleMixin, _BatchNorm):
    """Base class for Lazy Batch Normalization layers.

    This is an internal base class that implements lazy initialization for batch
    normalization. The number of features is automatically inferred from the first
    forward pass. This is useful when the number of input features is unknown at
    construction time.

    Args:
        momentum: Value used for running mean and variance computation. Default: 0.1
        eps: Value added to the denominator for numerical stability. Default: 1e-5
        affine: If ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, this module tracks the running mean and variance.
            Default: ``True``
        dtype: The desired data type of parameters and buffers. Default: None

    Attributes:
        weight: Uninitialized learnable weight parameter (initialized on first forward)
        bias: Uninitialized learnable bias parameter (initialized on first forward)
        running_mean: Uninitialized running mean buffer (initialized on first forward)
        running_var: Uninitialized running variance buffer (initialized on first forward)
        num_batches_tracked: Uninitialized batch counter (initialized on first forward)
    """

    running_mean: UninitializedBuffer
    running_var: UninitializedBuffer
    num_batches_tracked: UninitializedBuffer

    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(
        self,
        momentum=0.1,
        eps=0.00001,
        affine=True,
        track_running_stats=True,
        dtype=None,
    ):
        Module.__init__(self)

        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.affine = affine
        self.dtype = dtype
        self.eps = eps

        if self.track_running_stats:
            self.running_mean = UninitializedBuffer()
            self.running_var = UninitializedBuffer()
            self.num_batches_tracked = UninitializedBuffer()
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        if self.affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()
            self.weight.is_bn_param = True
            self.bias.is_bn_param = True
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        """Resets parameters if they have been initialized.

        This method only resets parameters after lazy initialization has occurred.
        Before initialization, this method does nothing.
        """
        if not self.has_uninitialized_params():
            super().reset_parameters()

    def initialize_parameters(self, input: Tensor) -> None:
        """Infers number of features from input and initializes all parameters.

        This method is called automatically on the first forward pass. It infers
        ``num_features`` from the channel dimension of the input tensor.

        Args:
            input: Input tensor used to infer the number of features
        """
        if self.has_uninitialized_params():
            with nova.no_grad():
                self.num_features = input.shape[1]
                if self.track_running_stats:

                    self.running_mean = self.running_mean.materialize(
                        (self.num_features,), dtype=self.dtype
                    )
                    self.running_var = self.running_var.materialize(
                        (self.num_features,), dtype=self.dtype
                    )
                    self.num_batches_tracked = self.num_batches_tracked.materialize(
                        (self.num_features,), dtype=nova.long
                    )

                if self.affine:
                    self.weight = self.weight.materialize(
                        (self.num_features,), dtype=self.dtype
                    )
                    self.bias = self.bias.materialize(
                        (self.num_features,), dtype=self.dtype
                    )

                self.reset_parameters()

    def extra_repr(self) -> str:
        if not hasattr(self, "num_features"):
            return f"num_features=?, momentum={self.momentum}, eps={self.eps}, affine={self.affine}, track_running_stats={self.track_running_stats}"
        return f"num_features={self.num_features}, momentum={self.momentum}, eps={self.eps}, affine={self.affine}, track_running_stats={self.track_running_stats}"


class BatchNorm1d(_BatchNorm):
    """Applies Batch Normalization over a 2D or 3D input.

    The input is expected to have shape :math:`(N, C)` or :math:`(N, C, L)`, where
    :math:`N` is the batch size, :math:`C` is the number of features or channels,
    and :math:`L` is the sequence length.

    This layer normalizes each feature independently across the batch dimension.

    .. math::
        y = \\frac{x - \\text{E}[x]}{\\sqrt{\\text{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard deviation are calculated over the :math:`N` and :math:`L`
    dimensions (if present) for each :math:`C` channel independently.

    Args:
        num_features: Number of features :math:`C` from an expected input of size
            :math:`(N, C)` or :math:`(N, C, L)`
        momentum: Value used for running mean and variance computation. Default: 0.1
        eps: Value added to the denominator for numerical stability. Default: 1e-5
        affine: If ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, this module tracks the running mean and variance.
            Default: ``True``
        dtype: The desired data type of parameters and buffers. Default: None

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # 2D input (batch_size, num_features)
        >>> m = BatchNorm1d(100)
        >>> x = nova.randn(20, 100)
        >>> y = m(x)
        >>> print(y.shape)
        (20, 100)

        >>> # 3D input (batch_size, num_features, sequence_length)
        >>> m = BatchNorm1d(100)
        >>> x = nova.randn(20, 100, 50)
        >>> y = m(x)
        >>> print(y.shape)
        (20, 100, 50)

        >>> # Without affine parameters
        >>> m = BatchNorm1d(100, affine=False)
        >>> print(m.weight)  # None
        >>> print(m.bias)    # None

        >>> # Without tracking running statistics
        >>> m = BatchNorm1d(100, track_running_stats=False)
        >>> # Uses batch statistics even during evaluation
    """

    def _check_input_dim(self, input: Tensor) -> None:
        """Validates that input is 2D or 3D.

        Args:
            input: Input tensor to validate

        Raises:
            ValueError: If input is not 2D or 3D
        """
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class LazyBatchNorm1d(_LayzyNormBase):
    """A BatchNorm1d layer with lazy initialization of the ``num_features`` argument.

    The number of features is inferred from the ``input.shape[1]`` on the first
    forward pass. This is useful when building models where the number of input
    features is unknown at construction time.

    Args:
        momentum: Value used for running mean and variance computation. Default: 0.1
        eps: Value added to the denominator for numerical stability. Default: 1e-5
        affine: If ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, this module tracks the running mean and variance.
            Default: ``True``
        dtype: The desired data type of parameters and buffers. Default: None

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # num_features is inferred on first forward
        >>> m = LazyBatchNorm1d()
        >>> x = nova.randn(20, 100)
        >>> y = m(x)
        >>> print(m.num_features)  # 100 (inferred from input)

        >>> # Works with 3D input as well
        >>> m = LazyBatchNorm1d()
        >>> x = nova.randn(20, 64, 50)
        >>> y = m(x)
        >>> print(m.num_features)  # 64
    """

    def _check_input_dim(self, input) -> None:
        """Validates that input is 2D or 3D.

        Args:
            input: Input tensor to validate

        Raises:
            ValueError: If input is not 2D or 3D
        """
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class BatchNorm2d(_BatchNorm):
    """Applies Batch Normalization over a 4D input.

    The input is expected to have shape :math:`(N, C, H, W)`, where :math:`N` is
    the batch size, :math:`C` is the number of channels, :math:`H` is the height,
    and :math:`W` is the width.

    This layer is commonly used after convolutional layers in CNNs. It normalizes
    each channel independently across the batch and spatial dimensions.

    .. math::
        y = \\frac{x - \\text{E}[x]}{\\sqrt{\\text{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard deviation are calculated over the :math:`N`, :math:`H`,
    and :math:`W` dimensions for each :math:`C` channel independently.

    Args:
        num_features: Number of channels :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        momentum: Value used for running mean and variance computation. Default: 0.1
        eps: Value added to the denominator for numerical stability. Default: 1e-5
        affine: If ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, this module tracks the running mean and variance.
            Default: ``True``
        dtype: The desired data type of parameters and buffers. Default: None

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # Standard usage after Conv2d
        >>> m = BatchNorm2d(64)
        >>> x = nova.randn(20, 64, 32, 32)
        >>> y = m(x)
        >>> print(y.shape)
        (20, 64, 32, 32)

        >>> # With custom momentum
        >>> m = BatchNorm2d(64, momentum=0.01)
        >>> # Slower adaptation to new statistics

        >>> # Training vs evaluation mode
        >>> m = BatchNorm2d(64)
        >>> m.train()  # Uses batch statistics
        >>> m.eval()   # Uses running statistics
    """

    def _check_input_dim(self, input) -> None:
        """Validates that input is 4D.

        Args:
            input: Input tensor to validate

        Raises:
            ValueError: If input is not 4D
        """
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class LazyBatchNorm2d(_LayzyNormBase):
    """A BatchNorm2d layer with lazy initialization of the ``num_features`` argument.

    The number of channels is inferred from the ``input.shape[1]`` on the first
    forward pass. This is particularly useful when building dynamic architectures
    where the number of channels is determined at runtime.

    Args:
        momentum: Value used for running mean and variance computation. Default: 0.1
        eps: Value added to the denominator for numerical stability. Default: 1e-5
        affine: If ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, this module tracks the running mean and variance.
            Default: ``True``
        dtype: The desired data type of parameters and buffers. Default: None

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # num_features is inferred on first forward
        >>> m = LazyBatchNorm2d()
        >>> x = nova.randn(20, 64, 32, 32)
        >>> y = m(x)
        >>> print(m.num_features)  # 64 (inferred from input)

        >>> # Useful in sequential models
        >>> model = Sequential(
        ...     LazyConv2d(64, 3),
        ...     LazyBatchNorm2d(),  # Automatically adapts to Conv2d output
        ...     ReLU()
        ... )
    """

    def _check_input_dim(self, input) -> None:
        """Validates that input is 4D.

        Args:
            input: Input tensor to validate

        Raises:
            ValueError: If input is not 4D
        """
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class BatchNorm3d(_BatchNorm):
    """Applies Batch Normalization over a 5D input.

    The input is expected to have shape :math:`(N, C, D, H, W)`, where :math:`N`
    is the batch size, :math:`C` is the number of channels, :math:`D` is the depth,
    :math:`H` is the height, and :math:`W` is the width.

    This layer is commonly used after 3D convolutional layers for video or volumetric
    data processing. It normalizes each channel independently across the batch and
    spatial dimensions.

    .. math::
        y = \\frac{x - \\text{E}[x]}{\\sqrt{\\text{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard deviation are calculated over the :math:`N`, :math:`D`,
    :math:`H`, and :math:`W` dimensions for each :math:`C` channel independently.

    Args:
        num_features: Number of channels :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        momentum: Value used for running mean and variance computation. Default: 0.1
        eps: Value added to the denominator for numerical stability. Default: 1e-5
        affine: If ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, this module tracks the running mean and variance.
            Default: ``True``
        dtype: The desired data type of parameters and buffers. Default: None

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # Standard usage after Conv3d for video processing
        >>> m = BatchNorm3d(64)
        >>> x = nova.randn(10, 64, 16, 32, 32)  # 10 videos, 64 channels, 16 frames
        >>> y = m(x)
        >>> print(y.shape)
        (10, 64, 16, 32, 32)

        >>> # For medical imaging (CT/MRI volumes)
        >>> m = BatchNorm3d(32)
        >>> x = nova.randn(4, 32, 64, 64, 64)  # 4 volumes, 32 channels
        >>> y = m(x)
    """

    def _check_input_dim(self, input) -> None:
        """Validates that input is 5D.

        Args:
            input: Input tensor to validate

        Raises:
            ValueError: If input is not 5D
        """
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")


class LazyBatchNorm3d(_LayzyNormBase):
    """A BatchNorm3d layer with lazy initialization of the ``num_features`` argument.

    The number of channels is inferred from the ``input.shape[1]`` on the first
    forward pass. This is useful for 3D architectures where the number of channels
    may be determined dynamically.

    Args:
        momentum: Value used for running mean and variance computation. Default: 0.1
        eps: Value added to the denominator for numerical stability. Default: 1e-5
        affine: If ``True``, this module has learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, this module tracks the running mean and variance.
            Default: ``True``
        dtype: The desired data type of parameters and buffers. Default: None

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # num_features is inferred on first forward
        >>> m = LazyBatchNorm3d()
        >>> x = nova.randn(10, 64, 16, 32, 32)
        >>> y = m(x)
        >>> print(m.num_features)  # 64 (inferred from input)

        >>> # Useful in 3D model architectures
        >>> model = Sequential(
        ...     LazyConv3d(64, 3),
        ...     LazyBatchNorm3d(),  # Automatically adapts to Conv3d output
        ...     ReLU()
        ... )
    """

    def _check_input_dim(self, input) -> None:
        """Validates that input is 5D.

        Args:
            input: Input tensor to validate

        Raises:
            ValueError: If input is not 5D
        """
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")
