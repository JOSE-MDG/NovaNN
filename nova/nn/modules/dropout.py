from __future__ import annotations
import nova.nn.functional as F
from typing import TYPE_CHECKING
from nova.nn.modules import Module

if TYPE_CHECKING:
    from nova import Tensor


class Dropout(Module):
    """Randomly zeroes elements of the input tensor with probability p during training.
    
    Dropout is a regularization technique that helps prevent overfitting by randomly
    dropping units (along with their connections) from the neural network during training.
    The dropped units are scaled by 1/(1-p) to maintain the expected sum.
    
    During evaluation (when ``training=False``), dropout does nothing and returns the input unchanged.
    
    .. math::
        \\text{output}_i = 
        \\begin{cases}
            0 & \\text{with probability } p \\\\
            \\frac{\\text{input}_i}{1-p} & \\text{with probability } 1-p
        \\end{cases}
    
    Args:
        p: Probability of an element to be zeroed. Must be in [0, 1). Default: 0.5
    
    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Output: :math:`(*)`, same shape as input
    
    Examples::
    
        >>> # Standard dropout for fully connected layers
        >>> dropout = Dropout(p=0.5)
        >>> x = nova.randn(32, 128, requires_grad=True)
        >>> output = dropout(x)
        >>> print(output.shape)  # (32, 128)
        
        >>> # During evaluation, dropout is disabled
        >>> dropout.eval()
        >>> output = dropout(x)
        >>> assert (output == x).all()  # No dropout applied
        
        >>> # Common usage in neural networks
        >>> class MyModel(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc1 = Linear(784, 256)
        ...         self.dropout = Dropout(0.5)
        ...         self.fc2 = Linear(256, 10)
        ...     
        ...     def forward(self, x):
        ...         x = relu(self.fc1(x))
        ...         x = self.dropout(x)  # Regularization
        ...         x = self.fc2(x)
        ...         return x
    
    Note:
        The scaling by 1/(1-p) ensures that the expected sum of the output
        remains the same as the input, which is important for maintaining
        stable training dynamics.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        """Applies dropout to the input tensor.

        Args:
            input: Input tensor of any shape.

        Returns:
            Tensor with dropout applied during training, unchanged during evaluation.
        """
        return F.dropout(input, self.p, self._training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Dropout2d(Module):
    """Randomly zeroes entire channels of the input tensor during training.
    
    Dropout2d is designed for 2D convolutional layers. Instead of dropping individual
    elements, it drops entire 2D feature maps (channels). This helps promote independence
    between feature maps. The dropped channels are scaled by 1/(1-p).
    
    Also known as Spatial Dropout or Feature Map Dropout in the literature.
    
    .. math::
        \\text{output}_{n,c,h,w} = 
        \\begin{cases}
            0 & \\text{if channel } c \\text{ is dropped} \\\\
            \\frac{\\text{input}_{n,c,h,w}}{1-p} & \\text{otherwise}
        \\end{cases}
    
    Args:
        p: Probability of a channel to be zeroed. Must be in [0, 1). Default: 0.5
    
    Shape:
        - Input: :math:`(N, C, H, W)` where N is batch size, C is channels,
          H is height, W is width
        - Output: :math:`(N, C, H, W)`, same shape as input
    
    Examples::
    
        >>> # Dropout for convolutional layers
        >>> dropout = Dropout2d(p=0.2)
        >>> x = nova.randn(8, 64, 32, 32, requires_grad=True)  # (N, C, H, W)
        >>> output = dropout(x)
        >>> print(output.shape)  # (8, 64, 32, 32)
        
        >>> # Usage in CNN architecture
        >>> class ConvNet(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = Conv2d(3, 64, kernel_size=3)
        ...         self.dropout1 = Dropout2d(0.2)
        ...         self.conv2 = Conv2d(64, 128, kernel_size=3)
        ...         self.dropout2 = Dropout2d(0.2)
        ...     
        ...     def forward(self, x):
        ...         x = relu(self.conv1(x))
        ...         x = self.dropout1(x)  # Drop entire channels
        ...         x = relu(self.conv2(x))
        ...         x = self.dropout2(x)
        ...         return x
        
        >>> # Entire feature maps are dropped together
        >>> x = nova.ones(1, 4, 8, 8)
        >>> dropout = Dropout2d(0.5)
        >>> dropout.train()
        >>> out = dropout(x)
        >>> # Some channels will be all zeros, others scaled by 1/(1-p)
    
    Note:
        This is preferred over standard Dropout for convolutional layers because
        adjacent pixels in feature maps are highly correlated. Dropping individual
        pixels is less effective than dropping entire channels.
    
    Reference:
        Tompson et al., "Efficient Object Localization Using Convolutional Networks"
        (2015) - Introduced spatial dropout for CNNs.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        """Applies channel-wise dropout to the input tensor.

        Args:
            input: Input tensor with shape (N, C, H, W).

        Returns:
            Tensor with entire channels dropped during training, unchanged during evaluation.

        Raises:
            ValueError: If input is not 4-dimensional.
        """
        return F.dropout2d(input, self.p, self._training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Dropout3d(Module):
    """Randomly zeroes entire channels of the 3D input tensor during training.
    
    Dropout3d is designed for 3D convolutional layers. It drops entire 3D feature maps
    (channels) rather than individual voxels. This is the 3D extension of Dropout2d,
    suitable for video data, 3D medical imaging, or volumetric CNNs.
    
    .. math::
        \\text{output}_{n,c,d,h,w} = 
        \\begin{cases}
            0 & \\text{if channel } c \\text{ is dropped} \\\\
            \\frac{\\text{input}_{n,c,d,h,w}}{1-p} & \\text{otherwise}
        \\end{cases}
    
    Args:
        p: Probability of a channel to be zeroed. Must be in [0, 1). Default: 0.5
    
    Shape:
        - Input: :math:`(N, C, D, H, W)` where N is batch size, C is channels,
          D is depth, H is height, W is width
        - Output: :math:`(N, C, D, H, W)`, same shape as input
    
    Examples::
    
        >>> # Dropout for 3D convolutional layers
        >>> dropout = Dropout3d(p=0.3)
        >>> x = nova.randn(4, 32, 16, 64, 64)  # (N, C, D, H, W)
        >>> output = dropout(x)
        >>> print(output.shape)  # (4, 32, 16, 64, 64)
        
        >>> # Usage in 3D CNN for video or medical imaging
        >>> class Conv3DNet(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = Conv3d(1, 32, kernel_size=3)
        ...         self.dropout1 = Dropout3d(0.2)
        ...         self.conv2 = Conv3d(32, 64, kernel_size=3)
        ...         self.dropout2 = Dropout3d(0.2)
        ...     
        ...     def forward(self, x):
        ...         x = relu(self.conv1(x))
        ...         x = self.dropout1(x)  # Drop entire 3D feature maps
        ...         x = relu(self.conv2(x))
        ...         x = self.dropout2(x)
        ...         return x
        
        >>> # Common for medical imaging (CT/MRI scans)
        >>> ct_scan = nova.randn(1, 1, 64, 256, 256)  # Single channel 3D volume
        >>> dropout = Dropout3d(0.1)
        >>> dropout.train()
        >>> regularized = dropout(ct_scan)
    
    Note:
        Like Dropout2d, this drops entire channels to account for the strong
        spatial correlation in 3D convolutional feature maps. Dropping individual
        voxels would be less effective.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        """Applies channel-wise dropout to the 3D input tensor.

        Args:
            input: Input tensor with shape (N, C, D, H, W).

        Returns:
            Tensor with entire 3D channels dropped during training, unchanged during evaluation.

        Raises:
            ValueError: If input is not 5-dimensional.
        """
        return F.dropout3d(input, self.p, self._training)

    def extra_repr(self) -> str:
        return f"p={self.p}"
