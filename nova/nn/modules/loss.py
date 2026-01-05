from __future__ import annotations
import nova.nn.functional as F
from typing import TYPE_CHECKING, Optional
from nova.nn.modules import Module

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import LossReducton


class MSELoss(Module):
    """Measures the mean squared error (squared L2 norm) between each element in the input and target.
    
    The loss can be described as:
    
    .. math::
        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
        l_n = (x_n - y_n)^2
    
    where :math:`N` is the batch size. If ``reduction`` is not ``'none'``, then:
    
    .. math::
        \\ell(x, y) =
        \\begin{cases}
            \\operatorname{mean}(L), & \\text{if reduction} = \\text{'mean';}\\\\
            \\operatorname{sum}(L),  & \\text{if reduction} = \\text{'sum'.}
        \\end{cases}
    
    MSE loss is commonly used for regression tasks where you want to minimize the
    squared difference between predictions and targets.
    
    Args:
        reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the sum of the output will be
            divided by the number of elements in the output, ``'sum'``: the output will be summed.
            Default: ``'mean'``
        weight: A manual rescaling weight given to each element. If given, has to be a Tensor of
            size matching the input. Default: None
            
    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Target: :math:`(*)`, same shape as the input
        - Output: Scalar if ``reduction`` is ``'mean'`` or ``'sum'``, otherwise :math:`(*)`
    
    Examples::
    
        >>> # Regression task
        >>> loss = MSELoss()
        >>> input = nova.randn(3, 5, requires_grad=True)
        >>> target = nova.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
        >>> print(output.item())  # Single scalar value
        
        >>> # Without reduction
        >>> loss = MSELoss(reduction='none')
        >>> input = nova.randn(3, 5)
        >>> target = nova.randn(3, 5)
        >>> output = loss(input, target)
        >>> print(output.shape)  # (3, 5)
        
        >>> # With sum reduction
        >>> loss = MSELoss(reduction='sum')
        >>> output = loss(input, target)
        >>> print(output.shape)  # ()
        
        >>> # With sample weights
        >>> weights = nova.tensor([1.0, 2.0, 3.0])
        >>> loss = MSELoss(weight=weights)
        >>> input = nova.randn(3, 5)
        >>> target = nova.randn(3, 5)
        >>> output = loss(input, target)
    """

    def __init__(
        self, reduction: LossReducton = "mean", weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the mean squared error loss.

        Args:
            input: Predicted values
            target: Ground truth values

        Returns:
            Computed loss value(s)
        """
        return F.mse_loss(input, target, self.weight, self.reduction)

    def extra_repr(self) -> str:
        return "reduction={reduction}".format(**self.__dict__)


class L1Loss(Module):
    """Measures the mean absolute error (MAE) between each element in the input and target.
    
    The loss can be described as:
    
    .. math::
        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
        l_n = |x_n - y_n|
    
    where :math:`N` is the batch size. If ``reduction`` is not ``'none'``, then:
    
    .. math::
        \\ell(x, y) =
        \\begin{cases}
            \\operatorname{mean}(L), & \\text{if reduction} = \\text{'mean';}\\\\
            \\operatorname{sum}(L),  & \\text{if reduction} = \\text{'sum'.}
        \\end{cases}
    
    L1 loss is less sensitive to outliers compared to MSE loss, making it more robust
    for regression tasks with noisy data.
    
    Args:
        reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            Default: ``'mean'``
        weight: A manual rescaling weight given to each element. Default: None
            
    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Target: :math:`(*)`, same shape as the input
        - Output: Scalar if ``reduction`` is ``'mean'`` or ``'sum'``, otherwise :math:`(*)`
    
    Examples::
    
        >>> # Basic usage for regression
        >>> loss = L1Loss()
        >>> input = nova.randn(3, 5, requires_grad=True)
        >>> target = nova.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
        
        >>> # More robust to outliers than MSE
        >>> input = nova.tensor([1.0, 2.0, 100.0])  # Contains outlier
        >>> target = nova.tensor([1.0, 2.0, 3.0])
        >>> l1 = L1Loss()(input, target)
        >>> mse = MSELoss()(input, target)
        >>> print(f"L1: {l1.item():.2f}, MSE: {mse.item():.2f}")  # L1 less affected
        
        >>> # Without reduction
        >>> loss = L1Loss(reduction='none')
        >>> output = loss(input, target)
        >>> print(output)  # Element-wise absolute differences
    """

    def __init__(
        self, reduction: LossReducton = "mean", weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the mean absolute error loss.

        Args:
            input: Predicted values
            target: Ground truth values

        Returns:
            Computed loss value(s)
        """
        return F.l1_loss(input, target, self.weight, self.reduction)

    def extra_repr(self) -> str:
        return "reduction={reduction}".format(**self.__dict__)


class SmoothL1Loss(Module):
    """Creates a criterion that uses a squared term if the absolute element-wise error falls below beta
    and an L1 term otherwise.
    
    Also known as the Huber loss, this loss combines the advantages of L1 and L2 losses. It is less
    sensitive to outliers than MSE loss and is smooth at the bottom, unlike L1 loss.
    
    .. math::
        \\text{loss}(x, y) = \\frac{1}{n} \\sum_{i=1}^{n} z_i
    
    where :math:`z_i` is given by:
    
    .. math::
        z_i = \\begin{cases}
            0.5 (x_i - y_i)^2 / \\text{beta}, & \\text{if } |x_i - y_i| < \\text{beta} \\\\
            |x_i - y_i| - 0.5 * \\text{beta}, & \\text{otherwise}
        \\end{cases}
    
    Args:
        reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            Default: ``'mean'``
        beta: Specifies the threshold at which to change between L1 and L2 loss.
            Default: 1.0
        weight: A manual rescaling weight given to each element. Default: None
            
    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Target: :math:`(*)`, same shape as the input
        - Output: Scalar if ``reduction`` is ``'mean'`` or ``'sum'``, otherwise :math:`(*)`
    
    Examples::
    
        >>> # Standard usage
        >>> loss = SmoothL1Loss()
        >>> input = nova.randn(3, 5, requires_grad=True)
        >>> target = nova.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
        
        >>> # Custom beta for different sensitivity
        >>> loss = SmoothL1Loss(beta=0.5)  # More sensitive to small errors
        >>> output = loss(input, target)
        
        >>> # Comparison with L1 and L2
        >>> input = nova.tensor([0.1, 1.0, 5.0])
        >>> target = nova.zeros(3)
        >>> smooth = SmoothL1Loss(beta=1.0)(input, target)
        >>> l1 = L1Loss()(input, target)
        >>> l2 = MSELoss()(input, target)
        >>> print(f"Smooth: {smooth:.2f}, L1: {l1:.2f}, L2: {l2:.2f}")
        
        >>> # Common in object detection (bounding box regression)
        >>> bbox_pred = nova.randn(32, 4)
        >>> bbox_target = nova.randn(32, 4)
        >>> loss = SmoothL1Loss()
        >>> bbox_loss = loss(bbox_pred, bbox_target)
    """

    def __init__(
        self,
        reduction: LossReducton = "mean",
        beta: float = 1.0,
        weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.beta = beta
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the smooth L1 loss.

        Args:
            input: Predicted values
            target: Ground truth values

        Returns:
            Computed loss value(s)
        """
        return F.smooth_l1_loss(input, target, self.beta, self.reduction, self.weight)

    def extra_repr(self) -> str:
        return "reduction={reduction}, beta={beta}".format(**self.__dict__)


class BCELoss(Module):
    """Measures the Binary Cross Entropy between the target and input probabilities.
    
    The loss can be described as:
    
    .. math::
        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
        l_n = - w_n \\left[ y_n \\cdot \\log x_n + (1 - y_n) \\cdot \\log (1 - x_n) \\right]
    
    where :math:`N` is the batch size. If ``reduction`` is not ``'none'``, then:
    
    .. math::
        \\ell(x, y) = \\begin{cases}
            \\operatorname{mean}(L), & \\text{if reduction} = \\text{'mean';}\\\\
            \\operatorname{sum}(L),  & \\text{if reduction} = \\text{'sum'.}
        \\end{cases}
    
    This loss is used for binary classification tasks. Note that the input should be
    probabilities (values between 0 and 1), typically from a sigmoid activation.
    
    Args:
        reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            Default: ``'mean'``
        weight: A manual rescaling weight given to the loss of each batch element.
            Default: None
            
    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Target: :math:`(*)`, same shape as the input. Values should be in [0, 1]
        - Output: Scalar if ``reduction`` is ``'mean'`` or ``'sum'``, otherwise :math:`(*)`
    
    Examples::
    
        >>> # Binary classification
        >>> m = Sigmoid()
        >>> loss = BCELoss()
        >>> input = nova.randn(3, requires_grad=True)
        >>> target = nova.tensor([0., 1., 1.])
        >>> output = loss(m(input), target)
        >>> output.backward()
        
        >>> # Multi-label classification
        >>> m = Sigmoid()
        >>> loss = BCELoss()
        >>> input = nova.randn(3, 4, requires_grad=True)
        >>> target = nova.tensor([[0., 1., 0., 1.],
        ...                        [1., 0., 1., 0.],
        ...                        [0., 0., 1., 1.]])
        >>> output = loss(m(input), target)
        
        >>> # With sample weights
        >>> weights = nova.tensor([1.0, 2.0, 1.5])
        >>> loss = BCELoss(weight=weights)
        >>> output = loss(m(input), target)
    
    Warning:
        This loss expects the input to be probabilities (between 0 and 1). If you have
        raw logits, use :class:`BCEWithLogitsLoss` instead for numerical stability.
    """

    def __init__(
        self,
        reduction: LossReducton = "mean",
        weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the binary cross entropy loss.

        Args:
            input: Predicted probabilities (after sigmoid), values in [0, 1]
            target: Ground truth binary labels, values in [0, 1]

        Returns:
            Computed loss value(s)
        """
        return F.binary_cross_entropy(input, target, self.weight, self.reduction)

    def extra_repr(self) -> str:
        return "reduction={reduction}".format(**self.__dict__)


class BCEWithLogitsLoss(Module):
    """Combines a Sigmoid layer and the BCELoss in one single class.

    This version is more numerically stable than using a plain Sigmoid followed by BCELoss
    as, by combining the operations into one layer, we take advantage of the log-sum-exp
    trick for numerical stability.

    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
        l_n = - w_n \\left[ y_n \\cdot \\log \\sigma(x_n) + (1 - y_n) \\cdot \\log (1 - \\sigma(x_n)) \\right]

    where :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-x)}` is the sigmoid function.

    Args:
        reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            Default: ``'mean'``
        weight: A manual rescaling weight given to the loss of each batch element.
            Default: None
        pos_weight: A weight of positive examples. Must be a vector with length equal to the
            number of classes. Useful for imbalanced datasets. Default: None

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions
        - Target: :math:`(*)`, same shape as the input. Values should be in [0, 1]
        - Output: Scalar if ``reduction`` is ``'mean'`` or ``'sum'``, otherwise :math:`(*)`

    Examples::

        >>> # Binary classification with raw logits
        >>> loss = BCEWithLogitsLoss()
        >>> input = nova.randn(3, requires_grad=True)
        >>> target = nova.tensor([0., 1., 1.])
        >>> output = loss(input, target)
        >>> output.backward()

        >>> # Multi-label classification
        >>> loss = BCEWithLogitsLoss()
        >>> input = nova.randn(3, 4, requires_grad=True)
        >>> target = nova.tensor([[0., 1., 0., 1.],
        ...                        [1., 0., 1., 0.],
        ...                        [0., 0., 1., 1.]])
        >>> output = loss(input, target)

        >>> # Imbalanced dataset with pos_weight
        >>> pos_weight = nova.tensor([2.0])  # Positive class weighted 2x
        >>> loss = BCEWithLogitsLoss(pos_weight=pos_weight)
        >>> input = nova.randn(10, 1)
        >>> target = nova.randint(0, 2, (10, 1)).float()
        >>> output = loss(input, target)

        >>> # Multi-label with different pos_weight per class
        >>> pos_weight = nova.tensor([1.0, 3.0, 2.0, 1.5])  # Per-class weights
        >>> loss = BCEWithLogitsLoss(pos_weight=pos_weight)
        >>> input = nova.randn(5, 4)
        >>> target = nova.randint(0, 2, (5, 4)).float()
        >>> output = loss(input, target)

    Note:
        This loss is preferred over ``BCELoss`` when working with raw logits for
        numerical stability reasons.
    """

    def __init__(
        self,
        reduction: LossReducton = "mean",
        weight: Optional[Tensor] = None,
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the binary cross entropy with logits loss.

        Args:
            input: Predicted logits (raw, unnormalized scores)
            target: Ground truth binary labels, values in [0, 1]

        Returns:
            Computed loss value(s)
        """
        return F.binary_cross_entropy_with_logits(
            input, target, self.weight, self.reduction, self.pos_weight
        )

    def extra_repr(self) -> str:
        return "reduction={reduction}".format(**self.__dict__)


class NLLLoss(Module):
    """The negative log likelihood loss.

    This loss is useful to train a classification problem with C classes. The input is
    expected to contain log-probabilities of each class (typically from LogSoftmax).

    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
        l_n = - w_{y_n} x_{n,y_n}, \\quad
        w_{c} = \\text{weight}[c] \\cdot \\mathbb{1}\\{c \\not= \\text{ignore\\_index}\\}

    where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
    and :math:`N` is the batch size.

    Args:
        reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            Default: ``'mean'``
        weight: A manual rescaling weight given to each class. If given, has to be a Tensor
            of size C (number of classes). Default: None

    Shape:
        - Input: :math:`(N, C)` where C = number of classes, or :math:`(N, C, d_1, d_2, ..., d_K)`
          with :math:`K \\geq 1` in the case of K-dimensional loss. Input should contain
          log-probabilities
        - Target: :math:`(N)` where each value is :math:`0 \\leq \\text{targets}[i] \\leq C-1`,
          or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of K-dimensional loss
        - Output: Scalar if ``reduction`` is ``'mean'`` or ``'sum'``, otherwise same shape as target

    Examples::

        >>> # Multi-class classification
        >>> m = LogSoftmax(dim=1)
        >>> loss = NLLLoss()
        >>> input = nova.randn(3, 5, requires_grad=True)
        >>> target = nova.tensor([1, 0, 4])
        >>> output = loss(m(input), target)
        >>> output.backward()

        >>> # With class weights for imbalanced dataset
        >>> weight = nova.tensor([1.0, 2.0, 1.5, 1.0, 3.0])
        >>> loss = NLLLoss(weight=weight)
        >>> output = loss(m(input), target)

        >>> # 2D target (e.g., semantic segmentation)
        >>> m = LogSoftmax(dim=1)
        >>> loss = NLLLoss()
        >>> input = nova.randn(2, 3, 10, 10, requires_grad=True)  # 2 images, 3 classes, 10x10
        >>> target = nova.randint(0, 3, (2, 10, 10))
        >>> output = loss(m(input), target)

    Note:
        This loss expects log-probabilities as input. Use with LogSoftmax, or use
        CrossEntropyLoss which combines LogSoftmax and NLLLoss for convenience.
    """

    def __init__(
        self, reduction: LossReducton = "mean", weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the negative log likelihood loss.

        Args:
            input: Predicted log-probabilities (typically from LogSoftmax)
            target: Ground truth class indices

        Returns:
            Computed loss value(s)
        """
        return F.nll_loss(input, target, self.weight, self.reduction)

    def extra_repr(self):
        return "reduction={reduction}".format(**self.__dict__)


class CrossEntropyLoss(Module):
    """Combines LogSoftmax and NLLLoss in one single class.

    This criterion computes the cross entropy loss between input logits and target class indices.
    It is useful for training a classification problem with C classes.

    The loss can be described as:

    .. math::
        \\text{loss}(x, \\text{class}) = -\\log\\left(\\frac{\\exp(x[\\text{class}])}{\\sum_j \\exp(x[j])}\\right)
                               = -x[\\text{class}] + \\log\\left(\\sum_j \\exp(x[j])\\right)

    This is the most commonly used loss for multi-class classification as it combines
    the softmax activation and negative log likelihood into a single, numerically stable
    operation.

    Args:
        reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            Default: ``'mean'``
        weight: A manual rescaling weight given to each class. If given, has to be a Tensor
            of size C (number of classes). Default: None

    Shape:
        - Input: :math:`(N, C)` where C = number of classes, or :math:`(N, C, d_1, d_2, ..., d_K)`
          with :math:`K \\geq 1` in the case of K-dimensional loss. Input should contain
          raw, unnormalized scores (logits)
        - Target: :math:`(N)` where each value is :math:`0 \\leq \\text{targets}[i] \\leq C-1`,
          or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of K-dimensional loss
        - Output: Scalar if ``reduction`` is ``'mean'`` or ``'sum'``, otherwise same shape as target

    Examples::

        >>> # Standard multi-class classification
        >>> loss = CrossEntropyLoss()
        >>> input = nova.randn(3, 5, requires_grad=True)
        >>> target = nova.tensor([1, 0, 4])
        >>> output = loss(input, target)
        >>> output.backward()

        >>> # With class weights for imbalanced dataset
        >>> weight = nova.tensor([1.0, 2.0, 1.5, 1.0, 3.0])
        >>> loss = CrossEntropyLoss(weight=weight)
        >>> input = nova.randn(10, 5, requires_grad=True)
        >>> target = nova.randint(0, 5, (10,))
        >>> output = loss(input, target)

        >>> # Image classification (e.g., CIFAR-10)
        >>> loss = CrossEntropyLoss()
        >>> logits = nova.randn(32, 10)  # 32 images, 10 classes
        >>> labels = nova.randint(0, 10, (32,))
        >>> output = loss(logits, labels)

        >>> # Semantic segmentation (per-pixel classification)
        >>> loss = CrossEntropyLoss()
        >>> input = nova.randn(2, 3, 10, 10, requires_grad=True)  # 2 images, 3 classes
        >>> target = nova.randint(0, 3, (2, 10, 10))
        >>> output = loss(input, target)

    Note:
        Unlike NLLLoss, this loss expects raw logits (unnormalized scores) as input,
        not log-probabilities. The LogSoftmax operation is performed internally for
        numerical stability.
    """

    def __init__(
        self, reduction: LossReducton = "mean", weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the cross entropy loss.

        Args:
            input: Predicted logits (raw, unnormalized scores)
            target: Ground truth class indices

        Returns:
            Computed loss value(s)
        """
        return F.cross_entropy(input, target, self.weight)

    def extra_repr(self):
        return "reduction={reduction}".format(**self.__dict__)


class KLDivLoss(Module):
    """The Kullback-Leibler divergence loss.

    Measures the divergence between two probability distributions. KL divergence is
    a useful distance measure for continuous distributions and is often used when
    performing direct regression over the space of (discretely sampled) continuous
    output distributions.

    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
        l_n = y_n \\cdot (\\log y_n - x_n)

    where :math:`x` is the input (log-probabilities), :math:`y` is the target
    (probabilities), and :math:`N` is the batch size.

    Args:
        reduction: Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'mean'``: the weighted mean of the output is taken, ``'sum'``: the output
            will be summed. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of dimensions. Should contain
          log-probabilities
        - Target: :math:`(*)`, same shape as the input. Should contain probabilities
        - Output: Scalar if ``reduction`` is ``'mean'`` or ``'sum'``, otherwise :math:`(*)`

    Examples::

        >>> # Measure divergence between two distributions
        >>> kl_loss = KLDivLoss()
        >>> input = nova.randn(3, 5)
        >>> input = F.log_softmax(input, dim=1)  # Input should be log-probabilities
        >>> target = nova.randn(3, 5)
        >>> target = F.softmax(target, dim=1)  # Target should be probabilities
        >>> output = kl_loss(input, target)

        >>> # Knowledge distillation (student learns from teacher)
        >>> teacher_logits = nova.randn(32, 10)
        >>> student_logits = nova.randn(32, 10)
        >>> temperature = 3.0
        >>> kl_loss = KLDivLoss(reduction='batchmean')
        >>> loss = kl_loss(
        ...     F.log_softmax(student_logits / temperature, dim=1),
        ...     F.softmax(teacher_logits / temperature, dim=1)
        ... ) * (temperature ** 2)

        >>> # Distribution matching
        >>> kl_loss = KLDivLoss(reduction='sum')
        >>> p = F.softmax(nova.randn(5), dim=0)
        >>> q = F.log_softmax(nova.randn(5), dim=0)
        >>> divergence = kl_loss(q, p)

    Note:
        The input should contain log-probabilities (typically from log_softmax), while
        the target should contain probabilities (typically from softmax). This is
        important for numerical stability.
    """

    def __init__(self, reduction: LossReducton = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the Kullback-Leibler divergence loss.

        Args:
            input: Predicted log-probabilities
            target: Target probabilities

        Returns:
            Computed loss value(s)
        """
        return F.kl_div(
            input,
            target,
            reduction=self.reduction,
        )

    def extra_repr(self):
        return "reduction={reduction}".format(**self.__dict__)
