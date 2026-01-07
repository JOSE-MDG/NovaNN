from __future__ import annotations
import nova
from nova.nn import init
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dtype, Dim


class ReLU(Module):
    """Applies the Rectified Linear Unit function element-wise.

    The ReLU activation function is defined as:

    .. math::
        \\text{ReLU}(x) = \\max(0, x)

    This operation sets all negative values in the input tensor to zero while
    keeping positive values unchanged. ReLU is one of the most widely used
    activation functions in deep learning due to its simplicity and effectiveness
    in preventing vanishing gradients.

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Output: :math:`(*)`, same shape as input

    Examples::

        >>> m = ReLU()
        >>> x = nova.tensor([-1.0, 0.0, 2.0])
        >>> y = m(x)
        >>> print(y)
        tensor([0.0, 0.0, 2.0])

        >>> # Works with multidimensional tensors
        >>> x = nova.tensor([[-1.0, 2.0], [3.0, -4.0]])
        >>> y = m(x)
        >>> print(y)
        tensor([[0.0, 2.0], [3.0, 0.0]])
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies ReLU activation element-wise.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor with same shape as input
        """
        return F.relu(input)


class LeakyReLU(Module):
    """Applies the Leaky Rectified Linear Unit function element-wise.
    
    The Leaky ReLU activation function is defined as:
    
    .. math::
        \\text{LeakyReLU}(x) = \\begin{cases}
            x, & \\text{if } x \\geq 0 \\\\
            \\text{negative\\_slope} \\times x, & \\text{otherwise}
        \\end{cases}
    
    Unlike standard ReLU, Leaky ReLU allows a small, non-zero gradient when
    the unit is not active. This helps prevent "dying ReLU" problem where
    neurons can become permanently inactive during training.
    
    Args:
        negative_slope: Controls the angle of the negative slope. Default: 0.01
        
    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Output: :math:`(*)`, same shape as input
    
    Examples::
    
        >>> m = LeakyReLU(negative_slope=0.1)
        >>> x = nova.tensor([-2.0, -1.0, 0.0, 3.0])
        >>> y = m(x)
        >>> print(y)
        tensor([-0.2, -0.1, 0.0, 3.0])
        
        >>> # Default negative_slope
        >>> m = LeakyReLU()
        >>> x = nova.tensor([-1.0, 1.0])
        >>> y = m(x)
        >>> print(y)
        tensor([-0.01, 1.0])
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        """Applies Leaky ReLU activation element-wise.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor with same shape as input
        """
        return F.leaky_relu(input, self.negative_slope)

    def extra_repr(self):
        return f"negative_slope={self.negative_slope}"


class GELU(Module):
    """Applies the Gaussian Error Linear Unit function element-wise.

    The GELU activation function is defined as:

    .. math::
        \\text{GELU}(x) = 0.5 \\times x \\times (1 + \\tanh(\\sqrt{2/\\pi} \\times (x + 0.044715 \\times x^3)))

    GELU is a smooth approximation to the ReLU that has been shown to work
    well in transformers and other modern architectures. It weights inputs
    by their value rather than by their sign.

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Output: :math:`(*)`, same shape as input

    Examples::

        >>> m = GeLU()
        >>> x = nova.tensor([-1.0, 0.0, 1.0])
        >>> y = m(x)
        >>> print(y)
        tensor([-0.1588, 0.0, 0.8411])

        >>> # GELU provides smooth gradients
        >>> x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> y = m(x)
        >>> print(y)
        tensor([-0.0454, -0.1588, 0.0, 0.8411, 1.9546])
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies GELU activation element-wise.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor with same shape as input
        """
        return F.gelu(input)


class PReLU(Module):
    """Applies the Parametric Rectified Linear Unit function element-wise.
    
    The PReLU activation function is defined as:
    
    .. math::
        \\text{PReLU}(x) = \\begin{cases}
            x, & \\text{if } x \\geq 0 \\\\
            a \\times x, & \\text{otherwise}
        \\end{cases}
    
    where :math:`a` is a learnable parameter. When :math:`a` is learned, it can
    have different values for different channels or be shared across all channels.
    
    Args:
        num_parameters: Number of :math:`a` to learn. Can be 1 (shared across all
            channels) or equal to the number of channels for channel-wise parameters.
            Default: 1
        init: Initial value for :math:`a`. Default: 0.25
        dtype: The desired data type of the parameter. Default: None
        
    Attributes:
        weight: The learnable weights of shape (num_parameters,)
        
    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Output: :math:`(*)`, same shape as input
    
    Examples::
    
        >>> # Single parameter for all channels
        >>> m = PReLU(num_parameters=1, init=0.25)
        >>> x = nova.tensor([-1.0, 0.0, 2.0])
        >>> y = m(x)
        >>> print(y)
        tensor([-0.25, 0.0, 2.0])
        >>> print(m.weight)
        Parameter containing: tensor([0.25])
        
        >>> # Channel-wise parameters
        >>> m = PReLU(num_parameters=3, init=0.1)
        >>> x = nova.tensor([[-1.0, -2.0, -3.0]])
        >>> y = m(x)
        >>> print(m.weight)
        Parameter containing: tensor([0.1, 0.1, 0.1])
    """

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        *,
        dtype: Optional[Dtype] = None,
    ) -> None:
        super().__init__()

        self.init = init
        self.num_parameters = num_parameters
        self.weight: Parameter = Parameter(
            nova.empty((num_parameters,)), dtype=nova.float32
        )

        self.reset_parameters()

    def reset_parameters(self):

        init.constant_(self.weight, self.init)

    def forward(self, input: Tensor) -> Tensor:
        """Applies PReLU activation using learnable weights.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor with same shape as input
        """
        return F.prelu(input, self.weight)

    def extra_repr(self):
        return f"num_parameters={self.num_parameters}"


class Tanh(Module):
    """Applies the Hyperbolic Tangent function element-wise.

    The Tanh activation function is defined as:

    .. math::
        \\text{Tanh}(x) = \\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}

    Tanh is a classic activation function that squashes input values to the
    range (-1, 1). It's zero-centered, making it easier for the model to learn.

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Output: :math:`(*)`, same shape as input

    Examples::

        >>> m = Tanh()
        >>> x = nova.tensor([-1.0, 0.0, 1.0])
        >>> y = m(x)
        >>> print(y)
        tensor([-0.7615, 0.0, 0.7615])

        >>> # Tanh outputs are bounded between -1 and 1
        >>> x = nova.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        >>> y = m(x)
        >>> print(y)
        tensor([-1.0, -0.7615, 0.0, 0.7615, 1.0])
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies Tanh activation element-wise.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor with same shape as input
        """
        return F.tanh(input)


class Sigmoid(Module):
    """Applies the Sigmoid function element-wise.

    The Sigmoid activation function is defined as:

    .. math::
        \\text{Sigmoid}(x) = \\sigma(x) = \\frac{1}{1 + e^{-x}}

    Sigmoid squashes input values to the range (0, 1), making it suitable
    for binary classification problems. However, it can suffer from vanishing
    gradients for very large or very small inputs.

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Output: :math:`(*)`, same shape as input

    Examples::

        >>> m = Sigmoid()
        >>> x = nova.tensor([-1.0, 0.0, 1.0])
        >>> y = m(x)
        >>> print(y)
        tensor([0.2689, 0.5, 0.7311])

        >>> # Sigmoid is commonly used for binary classification
        >>> x = nova.tensor([-5.0, 0.0, 5.0])
        >>> y = m(x)
        >>> print(y)
        tensor([0.0067, 0.5, 0.9933])
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Applies Sigmoid activation element-wise.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor with same shape as input
        """
        return F.sigmoid(input)


class Softmax(Module):
    """Applies the Softmax function along a specified dimension.

    The Softmax function is defined as:

    .. math::
        \\text{Softmax}(x_i) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}

    Softmax rescales elements to the range [0, 1] and ensures they sum to 1,
    making it ideal for multi-class classification problems where outputs
    represent class probabilities.

    Args:
        dim: Dimension along which Softmax will be computed. Default: 1

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Output: :math:`(*)`, same shape as input

    Examples::

        >>> m = Softmax(dim=0)
        >>> x = nova.tensor([1.0, 2.0, 3.0])
        >>> y = m(x)
        >>> print(y)
        tensor([0.0900, 0.2447, 0.6652])
        >>> print(y.sum())  # Probabilities sum to 1
        tensor(1.0)

        >>> # 2D example with dim=1
        >>> m = Softmax(dim=1)
        >>> x = nova.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        >>> y = m(x)
        >>> print(y)
        tensor([[0.0900, 0.2447, 0.6652],
                [0.3333, 0.3333, 0.3333]])
    """

    def __init__(self, dim: Dim = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        """Applies Softmax along the specified dimension.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor with same shape as input, with values in [0, 1]
            that sum to 1 along the specified dimension
        """
        return F.softmax(input, dim=self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"


class LogSoftmax(Module):
    """Applies the Log-Softmax function along a specified dimension.

    The LogSoftmax function is defined as:

    .. math::
        \\text{LogSoftmax}(x_i) = \\log\\left(\\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}\\right)

    LogSoftmax is numerically more stable than computing log(softmax(x))
    separately. It's commonly used in combination with Negative Log Likelihood
    loss for multi-class classification.

    Args:
        dim: Dimension along which LogSoftmax will be computed. Default: 1

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Output: :math:`(*)`, same shape as input

    Examples::

        >>> m = LogSoftmax(dim=0)
        >>> x = nova.tensor([1.0, 2.0, 3.0])
        >>> y = m(x)
        >>> print(y)
        tensor([-2.4076, -1.4076, -0.4076])

        >>> # LogSoftmax is more numerically stable
        >>> x = nova.tensor([[1.0, 2.0, 3.0]])
        >>> m = LogSoftmax(dim=1)
        >>> y = m(x)
        >>> print(y)
        tensor([[-2.4076, -1.4076, -0.4076]])
    """

    def __init__(self, dim: Dim = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        """Applies LogSoftmax along the specified dimension.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor with same shape as input containing log-probabilities
        """
        return F.log_softmax(input, dim=self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"
