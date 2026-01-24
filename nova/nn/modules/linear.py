from __future__ import annotations
import nova
import math
import nova.nn.init as init
import nova.nn.functional as F
from nova.nn.parameter import UninitializedParameter
from nova.nn.modules.lazy import LazyModuleMixin
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dtype


class Linear(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports TensorFloat32 (TF32) on Ampere and later CUDA devices.
    On Ampere and later CUDA devices, this module will use TF32 by default.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
        dtype: The desired data type of parameters. Default: None

    Attributes:
        weight: The learnable weights of the module of shape
            :math:`(\\text{out\\_features}, \\text{in\\_features})`. The values are
            initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
            :math:`k = \\frac{1}{\\text{in\\_features}}`
        bias: The learnable bias of the module of shape :math:`(1, \\text{out\\_features})`.
            If ``bias`` is ``True``, the values are initialized from
            :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
            :math:`k = \\frac{1}{\\text{in\\_features}}`

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of dimensions
          including none and :math:`H_{in} = \\text{in\\_features}`
        - Output: :math:`(*, H_{out})` where all but the last dimension are the same
          shape as the input and :math:`H_{out} = \\text{out\\_features}`

    Examples::

        >>> # Basic usage
        >>> m = Linear(20, 30)
        >>> x = nova.randn(128, 20)
        >>> output = m(x)
        >>> print(output.shape)
        (128, 30)

        >>> # Multi-dimensional input
        >>> m = Linear(20, 30)
        >>> x = nova.randn(10, 5, 20)
        >>> output = m(x)
        >>> print(output.shape)
        (10, 5, 30)

        >>> # Without bias
        >>> m = Linear(20, 30, bias=False)
        >>> print(m.bias)  # None
        >>> x = nova.randn(128, 20)
        >>> output = m(x)

        >>> # Building a simple MLP
        >>> model = Sequential(
        ...     Linear(784, 256),
        ...     ReLU(),
        ...     Linear(256, 128),
        ...     ReLU(),
        ...     Linear(128, 10)
        ... )

        >>> # Single sample (no batch dimension)
        >>> m = Linear(10, 5)
        >>> x = nova.randn(10)
        >>> output = m(x)
        >>> print(output.shape)
        (5,)
    """

    weight: Parameter
    bias: Optional[Parameter]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = Parameter(nova.empty((out_features, in_features)), dtype=dtype)
        if bias:
            self.bias = Parameter(nova.empty((1, out_features)), dtype=dtype)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets learnable parameters using Kaiming uniform initialization.

        Weight is initialized using Kaiming uniform initialization with a=sqrt(5),
        and bias (if present) is initialized uniformly within a range based on the
        fan-in to maintain variance across layers.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = init.get_fans(self.weight, mode="fan_in")
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """Applies the linear transformation to the input.

        Computes :math:`y = xA^T + b` where :math:`A` is the weight matrix and
        :math:`b` is the bias vector.

        Args:
            input: Input tensor of shape :math:`(*, H_{in})` where :math:`H_{in} = \\text{in\\_features}`

        Returns:
            Output tensor of shape :math:`(*, H_{out})` where :math:`H_{out} = \\text{out\\_features}`

        Examples::

            >>> m = Linear(20, 30)
            >>> x = nova.randn(128, 20)
            >>> y = m.forward(x)  # Equivalent to m(x)
            >>> print(y.shape)
            (128, 30)
        """
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class LazyLinear(LazyModuleMixin, Linear):
    """A Linear layer with lazy initialization of the ``in_features`` argument.

    The number of input features is inferred from the ``input.shape[-1]`` on the
    first forward pass. This is useful when building models where the input
    dimension is unknown at construction time, allowing for more flexible and
    dynamic model architectures.

    Unlike a regular Linear layer, LazyLinear only requires the output dimension
    to be specified upfront. The input dimension is automatically determined when
    the first batch of data passes through the layer.

    Args:
        out_features: Size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
        dtype: The desired data type of parameters. Default: None

    Attributes:
        weight: Uninitialized learnable weight parameter of shape
            :math:`(\\text{out\\_features}, \\text{in\\_features})` after first forward pass
        bias: Uninitialized learnable bias parameter of shape :math:`(1, \\text{out\\_features})`
            after first forward pass (if ``bias=True``)

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of dimensions
        - Output: :math:`(*, H_{out})` where :math:`H_{out} = \\text{out\\_features}`

    Examples::

        >>> # in_features is inferred on first forward
        >>> m = LazyLinear(30)
        >>> x = nova.randn(128, 20)
        >>> output = m(x)
        >>> print(m.in_features)  # 20 (inferred from input)
        >>> print(output.shape)
        (128, 30)

        >>> # Building flexible MLPs
        >>> model = Sequential(
        ...     LazyLinear(256),
        ...     ReLU(),
        ...     LazyLinear(128),  # Automatically adapts to 256 input features
        ...     ReLU(),
        ...     LazyLinear(10)    # Automatically adapts to 128 input features
        ... )
        >>> x = nova.randn(32, 784)  # Works with any input dimension
        >>> output = model(x)
        >>> print(output.shape)
        (32, 10)

        >>> # Multi-dimensional input
        >>> m = LazyLinear(50)
        >>> x = nova.randn(10, 5, 20)
        >>> output = m(x)
        >>> print(m.in_features)  # 20
        >>> print(output.shape)
        (10, 5, 50)

        >>> # Without bias
        >>> m = LazyLinear(30, bias=False)
        >>> x = nova.randn(128, 20)
        >>> output = m(x)
        >>> print(m.bias)  # None

        >>> # Check initialization status
        >>> m = LazyLinear(100)
        >>> print(m.has_uninitialized_params())  # True
        >>> x = nova.randn(32, 50)
        >>> _ = m(x)
        >>> print(m.has_uninitialized_params())  # False

    Note:
        After the first forward pass, LazyLinear behaves identically to a regular
        Linear layer. The lazy initialization only occurs once, after which all
        parameters are materialized with their correct shapes.
    """

    weight: UninitializedParameter
    bias: Optional[UninitializedParameter]

    def __init__(
        self, out_features: int, bias: bool = True, dtype: Optional[Dtype] = None
    ):
        Module.__init__(self)
        self.out_features = out_features
        self.use_bias = bias
        self.dtype = dtype

        self.weight = UninitializedParameter()
        if bias:
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        """Resets parameters if they have been initialized.

        This method only resets parameters after lazy initialization has occurred.
        Before initialization, this method does nothing. Once parameters are
        materialized, they are reset using the same initialization scheme as
        the regular Linear layer.
        """
        if not self.has_uninitialized_params():
            super().reset_parameters()

    def initialize_parameters(self, input: Tensor) -> None:
        """Infers ``in_features`` from input and initializes all parameters.

        This method is called automatically on the first forward pass. It infers
        the number of input features from the last dimension of the input tensor,
        materializes the weight and bias parameters with the correct shapes, and
        then initializes them using Kaiming uniform initialization.

        Args:
            input: Input tensor used to infer the number of input features from
                its last dimension

        Examples::

            >>> m = LazyLinear(30)
            >>> print(m.has_uninitialized_params())  # True
            >>> x = nova.randn(128, 20)
            >>> # initialize_parameters is called automatically
            >>> output = m(x)
            >>> print(m.in_features)  # 20
            >>> print(m.weight.shape)  # (30, 20)
            >>> print(m.has_uninitialized_params())  # False
        """
        if self.has_uninitialized_params():
            with nova.no_grad():
                self.in_features = input.shape[-1]
                self.weight = self.weight.materialize(
                    (self.out_features, self.in_features), dtype=self.dtype
                )
                if self.use_bias:
                    self.bias = self.bias.materialize(
                        (1, self.out_features), dtype=self.dtype
                    )
                self.reset_parameters()
