from __future__ import annotations
import nova
import nova.nn.init as init
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Size, Dtype


class LayerNorm(Module):
    """Applies Layer Normalization over the specified dimensions.

    Layer Normalization normalizes inputs across the feature dimension for each
    sample independently, unlike Batch Normalization which normalizes across the
    batch dimension. This makes LayerNorm particularly effective for:
    - Recurrent neural networks (RNNs, LSTMs, GRUs)
    - Transformers and attention mechanisms
    - Small batch sizes or online learning scenarios

    .. math::
        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated over the last D dimensions, where D
    is the dimension of ``normalized_shape``. For example, if ``normalized_shape`` is
    ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input.

    Args:
        normalized_shape: Input shape from an expected input of size.
            If a single integer, it is treated as a singleton tuple.
            If a tuple, normalization is performed over all dimensions in the tuple.
        eps: A value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: If True, adds learnable affine parameters (weight and bias).
            Default: True
        dtype: Data type for parameters. Default: None (uses default dtype)

    Attributes:
        weight: Learnable scale parameter of shape ``normalized_shape`` when
            ``elementwise_affine=True``. Initialized to ones.
        bias: Learnable shift parameter of shape ``normalized_shape`` when
            ``elementwise_affine=True``. Initialized to zeros.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means any number of dimensions.
          The last ``len(normalized_shape)`` dimensions must match ``normalized_shape``.
        - Output: :math:`(N, *)`, same shape as input

    Examples::

        >>> # Normalize over last dimension (common for transformers)
        >>> layer_norm = LayerNorm(512)
        >>> x = nova.randn(32, 10, 512)  # (batch, seq_len, features)
        >>> output = layer_norm(x)
        >>> print(output.shape)  # (32, 10, 512)

        >>> # Transformer layer with LayerNorm
        >>> class TransformerBlock(Module):
        ...     def __init__(self, d_model=512):
        ...         super().__init__()
        ...         self.attention = MultiHeadAttention(d_model)
        ...         self.norm1 = LayerNorm(d_model)
        ...         self.ffn = FeedForward(d_model)
        ...         self.norm2 = LayerNorm(d_model)
        ...
        ...     def forward(self, x):
        ...         # Pre-norm architecture
        ...         x = x + self.attention(self.norm1(x))
        ...         x = x + self.ffn(self.norm2(x))
        ...         return x

        >>> # Normalize over multiple dimensions
        >>> layer_norm = LayerNorm((3, 5))
        >>> x = nova.randn(20, 3, 5)
        >>> output = layer_norm(x)
        >>> print(output.shape)  # (20, 3, 5)

        >>> # Without learnable parameters
        >>> layer_norm = LayerNorm(128, elementwise_affine=False)
        >>> x = nova.randn(16, 10, 128)
        >>> output = layer_norm(x)
        >>> # Output is normalized but no learnable scaling/shifting

        >>> # Check normalization properties
        >>> x = nova.randn(8, 64)
        >>> ln = LayerNorm(64)
        >>> out = ln(x)
        >>> print(out.mean(dim=1))  # ~0 for each sample
        >>> print(out.var(dim=1))   # ~1 for each sample

    Note:
        Unlike BatchNorm, LayerNorm does not maintain running statistics and
        behaves the same during training and evaluation. This makes it more
        suitable for sequence models and scenarios with variable batch sizes.

    Reference:
        Ba et al., "Layer Normalization" (2016) - https://arxiv.org/abs/1607.06450
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        normalized_shape: Size,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(nova.empty((normalized_shape,)), dtype=dtype)
            self.bias = Parameter(nova.empty((normalized_shape,)), dtype=dtype)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes learnable parameters to their default values.

        Weight is initialized to ones, bias to zeros. This ensures the layer
        initially performs identity transformation after normalization.
        """
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """Applies layer normalization to the input.

        Args:
            input: Input tensor. The last ``len(normalized_shape)`` dimensions
                must match ``normalized_shape``.

        Returns:
            Normalized tensor with same shape as input.

        Raises:
            ValueError: If input shape doesn't match expected normalized_shape.
        """
        return F.layer_norm(
            input,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )

    def extra_repr(self) -> str:
        return "normalized_shape={normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )
