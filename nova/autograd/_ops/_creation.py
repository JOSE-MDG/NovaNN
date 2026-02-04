from __future__ import annotations
import nova
import numpy as np
import warnings
from typing import Any, Optional, TYPE_CHECKING
from nova.utils import ensure_tensor

if TYPE_CHECKING:
    from nova._typing import Dim, Dtype, Size, PaddingMode
[]
__all__ = [
    "sqrt",
    "mean",
    "var",
    "std",
    "empty",
    "min",
    "max",
    "sum",
    "pow",
    "maximum",
    "minimum",
    "clamp",
    "split",
    "tile",
    "as_strided",
    "repeat_interleave",
    "pad",
    "floor",
    "exp",
    "sin",
    "sinh",
    "cos",
    "cosh",
    "sec",
    "cot",
    "csc",
    "tan",
    "tanh",
    "atan2",
    "arcsin",
    "arccos",
    "arctan",
    "asinh",
    "acosh",
    "atanh",
    "arccot",
    "arcsec",
    "arccsc",
    "log",
    "isnan",
    "isinf",
    "reshape",
    "dot",
    "det",
    "inv",
    "trace",
    "norm",
    "abs",
    "sign",
    "where",
    "permute",
    "unsqueeze",
    "flatten",
    "cat",
    "eye",
    "one_hot",
    "full",
    "full_like",
    "arange",
    "unique",
    "argmin",
    "argmax",
    "argsort",
    "argwhere",
    "stack",
    "ceil",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "linspace",
    "logspace",
    "any",
    "all",
    "allclose",
]


def _resolve_padding_mode(mode: PaddingMode) -> str:
    MODES = ("zeros", "reflect", "replicate", "circular")
    match mode:
        case "zeros":
            mode = "constant"
        case "replicate":
            mode = "edge"
        case "reflect":
            mode = "reflect"
        case "circular":
            mode = "wrap"
        case _:
            raise ValueError(f"mode only accepts {MODES}, not '{mode}'")

    return mode


def sqrt(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise square root of the input tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor: A tensor containing the square roots of the elements in `input`.

    Examples:
        >>> x = nova.tensor([1.0, 4.0, 9.0], requires_grad=True)
        >>> y = nova.sqrt(x)
        >>> print(y)
        tensor([1.0, 2.0, 3.0], requires_grad=True)
    """
    input = ensure_tensor(input)
    return input.sqrt()


def mean(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    """
    Computes the mean of elements along a given dimension.

    Args:
        input (Tensor): Input tensor.
        dim (Dim, optional): Dimension along which to compute the mean.
        keepdims (bool, optional): If True, retains reduced dimensions.

    Returns:
        Tensor: The mean value(s) as a tensor.

    Examples:
        >>> x = nova.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        >>> y = nova.mean(x, dim=0)
        >>> print(y)
        tensor([2.0, 3.0], requires_grad=True)
    """
    input = ensure_tensor(input)
    return input.mean(dim, keepdims)


def var(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    """
    Computes the variance of elements along a given dimension.

    Args:
        input (Tensor): Input tensor.
        dim (Dim, optional): Dimension to reduce.
        keepdims (bool, optional): Whether to keep the reduced dimension.

    Returns:
        Tensor: The variance along the specified dimension.

    Examples:
        >>> x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = nova.var(x)
        >>> print(y)
        tensor(0.6666667, requires_grad=True)
    """
    input = ensure_tensor(input)
    return input.var(dim, keepdims)


def std(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    """
    Computes the standard deviation of elements along a given dimension.

    Args:
        input (Tensor): Input tensor.
        dim (Dim, optional): Dimension to reduce.
        keepdims (bool, optional): Whether to keep reduced dimension.

    Returns:
        Tensor: The standard deviation along the specified dimension.

    Examples:
        >>> x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = nova.std(x)
        >>> print(y)
        tensor(0.8164966, requires_grad=True)
    """
    input = ensure_tensor(input)
    return sqrt(input.var(dim, keepdims))


def empty(
    size: Size, *, dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    """
    Returns a new tensor filled with uninitialized values (zeros by default).

    Args:
        size (Size): Shape of the new tensor.
        dtype (Dtype, optional): Data type of the tensor.
        requires_grad (bool, optional): If True, enables gradient tracking.

    Returns:
        Tensor: A new tensor with specified properties.

    Examples:
        >>> x = nova.empty((2, 3))
        >>> print(x.shape)
        (2, 3)
    """
    dtype = dtype if dtype is not None else nova.float32

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data = np.empty(shape=size, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def min(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    """
    Returns the minimum value of all elements in the input tensor.

    Args:
        input (Tensor): Input tensor.
        dim (Dim, optional): Dimension to reduce.
        keepdims (bool, optional): Retain reduced dimensions.

    Returns:
        Tensor: The minimum value(s).

    Examples:
        >>> x = nova.tensor([[2.0, 3.0], [1.0, 4.0]])
        >>> y = nova.min(x)
        >>> print(y)
        tensor(1.0)
    """
    input = ensure_tensor(input)
    return input.min(dim=dim, keepdims=keepdims)


def max(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    """
    Returns the maximum value of all elements in the input tensor.

    Args:
        input (Tensor): Input tensor.
        dim (Dim, optional): Dimension to reduce.
        keepdims (bool, optional): Retain reduced dimensions.

    Returns:
        Tensor: The maximum value(s).

    Examples:
        >>> x = nova.tensor([[2.0, 3.0], [1.0, 4.0]])
        >>> y = nova.max(x)
        >>> print(y)
        tensor(4.0)
    """
    input = ensure_tensor(input)
    return input.max(dim=dim, keepdims=keepdims)


def sum(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    """
    Computes the sum of all elements in the input tensor.

    Args:
        input (Tensor): Input tensor.
        dim (Dim, optional): Dimension to reduce.
        keepdims (bool, optional): Retain reduced dimensions.

    Returns:
        Tensor: Sum of tensor elements.

    Examples:
        >>> x = nova.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        >>> y = nova.sum(x)
        >>> print(y)
        tensor(10.0, requires_grad=True)
    """
    input = ensure_tensor(input)
    return input.sum(dim=dim, keepdims=keepdims)


def pow(input: nova.Tensor, exponent: nova.Tensor | int | float) -> nova.Tensor:
    """
    Raises each element of the input tensor to the power of the given exponent.

    Args:
        input (Tensor): Base tensor.
        exponent (Tensor | int | float): Exponent to raise each element.

    Returns:
        Tensor: Result of the power operation.

    Examples:
        >>> x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = nova.pow(x, 2)
        >>> print(y)
        tensor([1.0, 4.0, 9.0], requires_grad=True)
    """
    input = ensure_tensor(input)
    exponent = ensure_tensor(exponent)
    return input.pow(exponent)


def maximum(input: nova.Tensor, other: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise maximum between two tensors.

    Args:
        input (Tensor): First input tensor.
        other (Tensor): Second input tensor.

    Returns:
        Tensor: A tensor with element-wise maxima.

    Examples:
        >>> x = nova.tensor([1.0, 5.0, 3.0])
        >>> y = nova.tensor([2.0, 4.0, 6.0])
        >>> z = nova.maximum(x, y)
        >>> print(z)
        tensor([2.0, 5.0, 6.0])
    """
    input = ensure_tensor(input)
    other = ensure_tensor(other)
    return input.maximum(other)


def minimum(input: nova.Tensor, other: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise minimum between two tensors.

    Args:
        input (Tensor): First input tensor.
        other (Tensor): Second input tensor.

    Returns:
        Tensor: A tensor containing the element-wise minima.

    Examples:
        >>> x = nova.tensor([1.0, 5.0, 3.0])
        >>> y = nova.tensor([2.0, 4.0, 6.0])
        >>> z = nova.minimum(x, y)
        >>> print(z)
        tensor([1.0, 4.0, 3.0])
    """
    input = ensure_tensor(input)
    other = ensure_tensor(other)
    return input.minimum(other)


def clamp(input: nova.Tensor, min_val: float, max_val: float) -> nova.Tensor:
    """
    Clamps all elements in the input tensor into the range [min_val, max_val].

    Args:
        input (Tensor): Input tensor.
        min_val (float): Lower bound of the range.
        max_val (float): Upper bound of the range.

    Returns:
        Tensor: Tensor with values clamped within [min_val, max_val].

    Examples:
        >>> x = nova.tensor([-1.0, 0.5, 3.0])
        >>> y = nova.clamp(x, 0.0, 1.0)
        >>> print(y)
        tensor([0.0, 0.5, 1.0])
    """
    input = ensure_tensor(input)
    return input.clamp(min_val, max_val)


def split(input: nova.Tensor, sections: int, dim: Dim = 0) -> nova.Tensor:
    """
    Splits the tensor into multiple sections along a given dimension.

    Args:
        input (Tensor): Tensor to split.
        sections (int): Number of sections to divide the tensor into.
        dim (Dim, optional): Dimension to split along.

    Returns:
        list[Tensor]: A list of tensors resulting from the split.

    Examples:
        >>> x = nova.tensor([1, 2, 3, 4, 5, 6])
        >>> parts = nova.split(x, 3)
        >>> [print(p) for p in parts]
        tensor([1, 2]
               [3, 4]
               [5, 6])
    """
    input = ensure_tensor(input)
    return input.split(sections=sections, dim=dim)


def tile(input: nova.Tensor, repeats: int) -> nova.Tensor:
    """
    Repeats the elements of the tensor along each dimension.

    Args:
        input (Tensor): Input tensor.
        repeats (int): Number of repetitions.

    Returns:
        Tensor: Tiled tensor.

    Examples:
        >>> x = nova.tensor([1, 2, 3])
        >>> y = nova.tile(x, 2)
        >>> print(y)
        tensor([1, 2, 3, 1, 2, 3])
    """
    input = ensure_tensor(input)
    return input.tile(repeats)


def repeat_interleave(input: nova.Tensor, repeats: int, dim: Optional[Dim] = None):
    """
    Repeats elements of a tensor along a given dimension.

    Args:
        input (Tensor): Input tensor.
        repeats (int): Number of repetitions per element.
        dim (int): Dimension along which to repeat.

    Returns:
        Tensor: Tensor with repeated elements.

    Examples:
        >>> x = nova.tensor([[1, 2], [3, 4]])
        >>> y = nova.repeat_interleave(x, 2, dim=1)
        >>> print(y)
        tensor([[1, 1, 2, 2],
                [3, 3, 4, 4]])
    """
    input = ensure_tensor(input)
    return input.repeat(repeats=repeats, dim=dim)


def pad(
    input: nova.Tensor,
    pad_width: tuple[tuple[int, ...], ...] | tuple[int, ...],
    mode: PaddingMode = "zeros",
):
    """
    Pads the tensor with a specified mode and width.

    Args:
        input (Tensor): Input tensor.
        pad_width (tuple): Tuple specifying padding widths per dimension.
        mode (str, optional): Padding mode ("zeros", "reflect", "replicate", "circular").

    Returns:
        Tensor: Padded tensor.

    Examples:
        >>> x = nova.tensor([[1, 2], [3, 4]])
        >>> y = nova.pad(x, ((1, 1), (1, 1)), mode="zeros")
        >>> print(y)
        tensor([[0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0]])
    """
    input = ensure_tensor(input)

    mode = _resolve_padding_mode(mode)

    return input.pad(pad_width, mode)


def floor(input: nova.Tensor):
    """
    Applies the floor function element-wise to the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with floored values.

    Examples:
        >>> x = nova.tensor([1.7, -2.3, 3.9])
        >>> y = nova.floor(x)
        >>> print(y)
        tensor([1.0, -3.0, 3.0])
    """
    input = ensure_tensor(input)
    return input.floor()


def ceil(input: nova.Tensor):
    """
    Applies the ceil function element-wise to the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with ceiled values.

    Examples:
        >>> x = nova.tensor([1.2, -2.8, 3.1])
        >>> y = nova.ceil(x)
        >>> print(y)
        tensor([2.0, -2.0, 4.0])
    """
    input = ensure_tensor(input)
    return input.ceil()


def exp(input: nova.Tensor) -> nova.Tensor:
    """
    Applies the exponential function element-wise.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with exponentiated values.

    Examples:
        >>> x = nova.tensor([0.0, 1.0, 2.0])
        >>> y = nova.exp(x)
        >>> print(y)
        tensor([1.0, 2.718, 7.389])
    """
    input = ensure_tensor(input)
    return input.exp()


def sin(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise sine of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing the sine of each element.

    Examples:
        >>> x = nova.tensor([0.0, np.pi / 2])
        >>> y = nova.sin(x)
        >>> print(y)
        tensor([0.0, 1.0])
    """
    input = ensure_tensor(input)
    return input.sin()


def sinh(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise hyperbolic sine of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing the hyperbolic sine of each element.

    Examples:
        >>> x = nova.tensor([0.0, 1.0, -1.0])
        >>> y = nova.sinh(x)
        >>> print(y)
        tensor([0.0, 1.1752, -1.1752])
    """
    input = ensure_tensor(input)
    return input.sinh()


def cos(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise cosine of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing the cosine of each element.

    Examples:
        >>> x = nova.tensor([0.0, np.pi])
        >>> y = nova.cos(x)
        >>> print(y)
        tensor([1.0, -1.0])
    """
    input = ensure_tensor(input)
    return input.cos()


def cosh(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise hyperbolic cosine of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing the hyperbolic cosine of each element.

    Examples:
        >>> x = nova.tensor([0.0, 1.0, -1.0])
        >>> y = nova.cosh(x)
        >>> print(y)
        tensor([1.0, 1.5431, 1.5431])
    """
    input = ensure_tensor(input)
    return input.cosh()


def tan(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise tangent of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing tangent of each element.

    Examples:
        >>> x = nova.tensor([0.0, np.pi / 4])
        >>> y = nova.tan(x)
        >>> print(y)
        tensor([0.0, 1.0])
    """
    input = ensure_tensor(input)
    return input.tan()


def tanh(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise hyperbolic tangent of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing tanh of each element.

    Examples:
        >>> x = nova.tensor([-1.0, 0.0, 1.0])
        >>> y = nova.tanh(x)
        >>> print(y)
        tensor([-0.7615, 0.0, 0.7615])
    """
    input = ensure_tensor(input)
    return input.tanh()


def sec(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise secant (1 / cos) of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing secant of each element.

    Examples:
        >>> x = nova.tensor([0.0, np.pi / 4])
        >>> y = nova.sec(x)
        >>> print(y)
        tensor([1.0, 1.4142])
    """
    input = ensure_tensor(input)
    return input.sec()


def cot(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise cotangent (1 / tan) of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing cotangent of each element.

    Examples:
        >>> x = nova.tensor([np.pi / 4, np.pi / 2])
        >>> y = nova.cot(x)
        >>> print(y)
        tensor([1.0, 0.0])
    """
    input = ensure_tensor(input)
    return input.cot()


def csc(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise cosecant (1 / sin) of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing cosecant of each element.

    Examples:
        >>> x = nova.tensor([np.pi / 2, np.pi / 6])
        >>> y = nova.csc(x)
        >>> print(y)
        tensor([1.0, 2.0])
    """
    input = ensure_tensor(input)
    return input.csc()


def atan2(input: nova.Tensor, other: nova.Tensor | float | int) -> nova.Tensor:
    """
    Computes the element-wise arctangent of input/other, choosing the quadrant correctly.

    Args:
        input (Tensor): Y-coordinates (numerator).
        other (Tensor | float | int): X-coordinates (denominator).

    Returns:
        Tensor: Tensor containing arctangent values in radians, range [-π, π].

    Examples:
        >>> y = nova.tensor([1.0, -1.0, 1.0])
        >>> x = nova.tensor([1.0, 1.0, -1.0])
        >>> z = nova.atan2(y, x)
        >>> print(z)
        tensor([0.7854, -0.7854, 2.3562])
    """
    input = ensure_tensor(input)
    other = ensure_tensor(other)  # This method handles it correctly
    return input.atan2(other)


def arcsin(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the inverse sine of each element (in radians).

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with arcsin values.

    Examples:
        >>> x = nova.tensor([0.0, 1.0])
        >>> y = nova.arcsin(x)
        >>> print(y)
        tensor([0.0, 1.5708])
    """
    input = ensure_tensor(input)
    return input.arcsin()


def arccos(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the inverse cosine of each element (in radians).

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with arccos values.

    Examples:
        >>> x = nova.tensor([1.0, 0.0])
        >>> y = nova.arccos(x)
        >>> print(y)
        tensor([0.0, 1.5708])
    """
    input = ensure_tensor(input)
    return input.arccos()


def arctan(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the inverse tangent of each element (in radians).

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with arctangent values.

    Examples:
        >>> x = nova.tensor([0.0, 1.0])
        >>> y = nova.arctan(x)
        >>> print(y)
        tensor([0.0, 0.7854])
    """
    input = ensure_tensor(input)
    return input.arctan()


def asinh(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise inverse hyperbolic sine of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing inverse hyperbolic sine of each element.

    Examples:
        >>> x = nova.tensor([0.0, 1.0, -1.0])
        >>> y = nova.asinh(x)
        >>> print(y)
        tensor([0.0, 0.8814, -0.8814])
    """
    input = ensure_tensor(input)
    return input.asinh()


def acosh(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise inverse hyperbolic cosine of the input tensor.

    Args:
        input (Tensor): Input tensor (values must be >= 1).

    Returns:
        Tensor: Tensor containing inverse hyperbolic cosine of each element.

    Examples:
        >>> x = nova.tensor([1.0, 2.0, 3.0])
        >>> y = nova.acosh(x)
        >>> print(y)
        tensor([0.0, 1.3170, 1.7627])
    """
    input = ensure_tensor(input)
    return input.acosh()


def atanh(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise inverse hyperbolic tangent of the input tensor.

    Args:
        input (Tensor): Input tensor (values must be in range (-1, 1)).

    Returns:
        Tensor: Tensor containing inverse hyperbolic tangent of each element.

    Examples:
        >>> x = nova.tensor([0.0, 0.5, -0.5])
        >>> y = nova.atanh(x)
        >>> print(y)
        tensor([0.0, 0.5493, -0.5493])
    """
    input = ensure_tensor(input)
    return input.atanh()


def arccot(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise inverse cotangent of the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor containing inverse cotangent of each element.

    Examples:
        >>> x = nova.tensor([1.0, 2.0, -1.0])
        >>> y = nova.arccot(x)
        >>> print(y)
        tensor([0.7854, 0.4636, -0.7854])
    """
    input = ensure_tensor(input)
    return input.arccot()


def arcsec(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise inverse secant of the input tensor.

    Args:
        input (Tensor): Input tensor (|values| must be >= 1).

    Returns:
        Tensor: Tensor containing inverse secant of each element.

    Examples:
        >>> x = nova.tensor([1.0, 2.0, -2.0])
        >>> y = nova.arcsec(x)
        >>> print(y)
        tensor([0.0, 1.0472, 2.0944])
    """
    input = ensure_tensor(input)
    return input.arcsec()


def arccsc(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the element-wise inverse cosecant of the input tensor.

    Args:
        input (Tensor): Input tensor (|values| must be >= 1).

    Returns:
        Tensor: Tensor containing inverse cosecant of each element.

    Examples:
        >>> x = nova.tensor([1.0, 2.0, -2.0])
        >>> y = nova.arccsc(x)
        >>> print(y)
        tensor([1.5708, 0.5236, -0.5236])
    """
    input = ensure_tensor(input)
    return input.arccsc()


def log(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the natural logarithm element-wise.

    Args:
        input (Tensor): Input tensor (positive values only).

    Returns:
        Tensor: Tensor containing the logarithm of each element.

    Examples:
        >>> x = nova.tensor([1.0, np.e, np.e ** 2])
        >>> y = nova.log(x)
        >>> print(y)
        tensor([0.0, 1.0, 2.0])
    """
    input = ensure_tensor(input)
    return input.log()


def isnan(input: nova.Tensor) -> np.ndarray:
    """
    Returns a boolean array indicating if each element is NaN.

    Args:
        input (Tensor): Input tensor.

    Returns:
        np.ndarray: A boolean array of the same shape as input.

    Examples:
        >>> x = nova.tensor([1.0, np.nan, 2.0])
        >>> print(nova.isnan(x))
        [False  True False]
    """
    input = ensure_tensor(input)
    return np.isnan(input.data)


def isinf(input: nova.Tensor) -> np.ndarray:
    """
    Returns a boolean array indicating if each element is positive or negative infinity.

    Args:
        input (Tensor): Input tensor.

    Returns:
        np.ndarray: A boolean array of the same shape as input.

    Examples:
        >>> x = nova.tensor([1.0, np.inf, -np.inf])
        >>> print(nova.isinf(x))
        [False  True  True]
    """
    input = ensure_tensor(input)
    return np.isinf(input.data)


def reshape(input: nova.Tensor, dims: Dim) -> nova.Tensor:
    """
    Returns a tensor with the same data but different shape.

    Args:
        input (Tensor): Input tensor.
        dims (Dim): New shape.

    Returns:
        Tensor: Reshaped tensor.

    Examples:
        >>> x = nova.tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = nova.reshape(x, (3, 2))
        >>> print(y)
        tensor([[1, 2],
                [3, 4],
                [5, 6]])
    """
    input = ensure_tensor(input)
    return input.reshape(*dims)


def dot(input: nova.Tensor, other: nova.Tensor) -> nova.Tensor:
    """
    Computes the dot product between two tensors.

    Args:
        input (Tensor): First input tensor.
        other (Tensor): Second input tensor.

    Returns:
        Tensor: Scalar tensor representing the dot product.

    Examples:
        >>> x = nova.tensor([1.0, 2.0, 3.0])
        >>> y = nova.tensor([4.0, 5.0, 6.0])
        >>> z = nova.dot(x, y)
        >>> print(z)
        tensor(32.0)
    """
    input = ensure_tensor(input)
    other = ensure_tensor(other)
    return input.dot(other)


def det(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the determinant of a square matrix tensor.

    Args:
        input (Tensor): Square matrix tensor.

    Returns:
        Tensor: Scalar tensor representing the determinant.

    Examples:
        >>> x = nova.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> y = nova.det(x)
        >>> print(y)
        tensor(-2.0)
    """
    input = ensure_tensor(input)
    return input.det()


def inv(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the inverse of a square matrix tensor.

    Args:
        input (Tensor): Square matrix tensor.

    Returns:
        Tensor: Inverse matrix tensor.

    Examples:
        >>> x = nova.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> y = nova.inv(x)
        >>> print(y)
        tensor([[-2.0, 1.0],
                [1.5, -0.5]])
    """
    input = ensure_tensor(input)
    return input.inv()


def trace(input: nova.Tensor) -> nova.Tensor:
    """
    Computes the trace (sum of diagonal elements) of a matrix tensor.

    Args:
        input (Tensor): Matrix tensor.

    Returns:
        Tensor: Scalar tensor representing the trace.

    Examples:
        >>> x = nova.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> y = nova.trace(x)
        >>> print(y)
        tensor(5.0)
    """
    input = ensure_tensor(input)
    return input.trace()


def norm(
    input: nova.Tensor, ord: int = 2, dim: Optional[Dim] = None, keepdims: bool = False
) -> nova.Tensor:
    """
    Computes the matrix or vector norm.

    Args:
        input (Tensor): Input tensor.
        ord (int, optional): Order of the norm (default 2).
        dim (Dim, optional): Dimension to compute norm over.
        keepdims (bool, optional): Whether to retain reduced dimensions.

    Returns:
        Tensor: Tensor representing the computed norm.

    Examples:
        >>> x = nova.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> y = nova.norm(x)
        >>> print(y)
        tensor(5.4772)
    """
    input = ensure_tensor(input)
    return input.norm(ord, dim=dim, keepdims=keepdims)


def abs(input: nova.Tensor) -> nova.Tensor:
    """
    Returns the absolute value of each element in the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with absolute values.

    Examples:
        >>> x = nova.tensor([-1.0, 0.0, 2.0])
        >>> y = nova.abs(x)
        >>> print(y)
        tensor([1.0, 0.0, 2.0])
    """
    input = ensure_tensor(input)
    return input.abs()


def sign(input: nova.Tensor) -> nova.Tensor:
    """
    Returns the sign of each element in the tensor:
    -1 for negative, 0 for zero, 1 for positive.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Tensor of signs.

    Examples:
        >>> x = nova.tensor([-3.0, 0.0, 5.0])
        >>> y = nova.sign(x)
        >>> print(y)
        tensor([-1.0, 0.0, 1.0])
    """
    input = ensure_tensor(input)
    return input.sign()


def where(condition: nova.Tensor, x: nova.Tensor, y: nova.Tensor):
    """
    Selects elements from `x` or `y` depending on `condition`.

    Args:
        condition (Tensor): Boolean condition tensor.
        x (Tensor): Values selected where condition is True.
        y (Tensor): Values selected where condition is False.

    Returns:
        Tensor: Tensor composed of elements from `x` or `y`.

    Examples:
        >>> cond = nova.tensor([True, False, True])
        >>> x = nova.tensor([1, 2, 3])
        >>> y = nova.tensor([9, 9, 9])
        >>> z = nova.where(cond, x, y)
        >>> print(z)
        tensor([1, 9, 3])
    """
    from nova.autograd._ops import Where

    x = ensure_tensor(x)
    y = ensure_tensor(y)
    return Where.apply(condition, x, y)


def permute(input: nova.Tensor, *dims: Dim) -> nova.Tensor:
    """
    Permutes the dimensions of the input tensor according to `dims`.

    Args:
        input (Tensor): Input tensor.
        *dims (Dim): Desired ordering of dimensions.

    Returns:
        Tensor: Permuted tensor.

    Examples:
        >>> x = nova.tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = nova.permute(x, 1, 0)
        >>> print(y)
        tensor([[1, 4],
                [2, 5],
                [3, 6]])
    """
    input = ensure_tensor(input)
    return input.permute(*dims)


def unsqueeze(input: nova.Tensor, dim: Dim) -> nova.Tensor:
    """
    Adds a dimension of size one at the specified position.

    Args:
        input (Tensor): Input tensor.
        dim (Dim): Position at which to insert the new dimension.

    Returns:
        Tensor: Tensor with an extra dimension.

    Examples:
        >>> x = nova.tensor([1, 2, 3])
        >>> y = nova.unsqueeze(x, 0)
        >>> print(y.shape)
        (1, 3)
    """
    input = ensure_tensor(input)
    return input.unsqueeze(dim)


def flatten(input: nova.Tensor, start_dim: int = 0, end_dim: int = -1) -> nova.Tensor:
    """
    Flattens continuous dimensions within a range.

    Args:
        input (Tensor): Tensor to be flattened
        start_dim (int): First dimension to flatten (default: 0)
        end_dim (int): Last dimension to flatten (default: -1, last dim)

    Returns:
        Tensor with flattened dimensions
    Examples:
        >>> import nova
        >>> x = nova.randn(2, 3, 4, 5)
        >>> nova.flatten(x, 1, 2).shape
        (2, 12, 5)  # flattened dims 1 and 2: 3*4=12

        >>> nova.flatten(x, 0, -1).shape
        (120,)  # all dimensions flattened: 2*3*4*5=120

        >>> nova.flatten(x, 1).shape
        (2, 60)  # from dim 1 to the end: 3*4*5=60
    """
    input = ensure_tensor(input)
    return input.flatten(start_dim, end_dim)


def cat(inputs: list[nova.Tensor], dim: Dim = None):
    """
    Concatenates a list of tensors along a given dimension.

    Args:
        inputs (list[Tensor]): List of tensors to concatenate.
        dim (Dim, optional): Dimension to concatenate along.

    Returns:
        Tensor: Concatenated tensor.

    Examples:
        >>> a = nova.tensor([[1, 2]])
        >>> b = nova.tensor([[3, 4]])
        >>> c = nova.cat([a, b], dim=0)
        >>> print(c)
        tensor([[1, 2],
                [3, 4]])
    """
    from nova.autograd._ops import Concat

    return Concat.apply(inputs, dim)


def eye(
    N: int,
    M: Optional[int] = None,
    K: int = 0,
    dtype: Optional[Dtype] = None,
    requires_grad: bool = False,
) -> nova.Tensor:
    """
    Creates a 2-D identity matrix with ones on the diagonal and zeros elsewhere.

    Args:
        N (int): Number of rows.
        M (Optional[int]): Number of columns. Defaults to N if not provided.
        K (int): Index of the diagonal. Defaults to 0.
        dtype (Optional[Dtype]): Data type of the returned tensor.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: Identity matrix of shape (N, M).

    Examples:
        >>> x = nova.eye(3)
        >>> print(x)
        tensor([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    """
    if M is None:
        M = N

    data = np.eye(N, M, k=K, dtype=dtype if dtype is not None else nova.float32)
    return nova.Tensor(
        data,
        dtype=dtype if dtype is not None else nova.float32,
        requires_grad=requires_grad,
    )


def one_hot(labels: nova.Tensor, num_classes: int) -> nova.Tensor:
    """
    Converts a tensor of labels to one-hot encoded format.

    Args:
        labels (Tensor): Input tensor with integer class labels.
        num_classes (int): Total number of classes.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: One-hot encoded tensor.

    Examples:
        >>> labels = nova.tensor([0, 2, 1])
        >>> y = nova.one_hot(labels, num_classes=3)
        >>> print(y)
        tensor([[1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]])
    """
    if not isinstance(num_classes, int):
        raise ValueError(f"num_classes expect a integer, got '{type(num_classes)}'")
    labels = ensure_tensor(labels, dtype=nova.long)

    data = nova.eye(num_classes)[labels]
    return data


def full(
    size: Size,
    fill_value: Any,
    *,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
) -> nova.Tensor:
    """
    Creates a tensor filled with the specified value.

    Args:
        size (Size | list): Shape of the output tensor.
        fill_value (Any): Value to fill the tensor with.
        dtype (Optional[Dtype]): Data type of the tensor.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: Tensor filled with `fill_value`.

    Examples:
        >>> x = nova.full((2,3), 7)
        >>> print(x)
        tensor([[7, 7, 7],
                [7, 7, 7]])
    """

    if dtype is None:
        dtype = nova.float32

    data = np.full(
        shape=size,
        fill_value=fill_value,
        dtype=dtype,
    )
    return nova.Tensor(
        data,
        dtype=dtype,
        requires_grad=requires_grad,
    )


def full_like(
    input: nova.Tensor,
    fill_value: Any,
    *,
    dtype: Optional[Dtype] = None,
    requires_grad: bool = False,
) -> nova.Tensor:
    """
    Returns a tensor with the same size as `input` filled with `fill_value`.

    Args:
        input (Tensor): The reference tensor whose shape will be mimicked.
        fill_value (Any): Value to fill the output tensor with.
        dtype (Optional[Dtype]): Data type of the tensor. If None, it defaults
            to nova.float32.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: A tensor of the same shape as `input` filled with `fill_value`.

    Examples:
        >>> x = nova.ones((2, 2))
        >>> y = nova.full_like(x, 3.14)
        >>> print(y)
        tensor([[3.14, 3.14],
                [3.14, 3.14]])
    """

    input = ensure_tensor(input)

    if dtype is None:
        dtype = nova.float32

    data = np.full_like(input.data, fill_value=fill_value, dtype=dtype)
    return nova.Tensor(data, dtype=dtype, requires_grad=requires_grad)


def arange(
    start: int | float,
    stop: Optional[int | float] = None,
    step: int | float = 1,
    *,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
) -> nova.Tensor:
    """
    Returns a 1-D tensor with values from `start` to `stop` (exclusive) with a given step.

    Args:
        start (int | float): Start of the interval.
        stop (Optional[int | float]): End of the interval. If None, interval is [0, start).
        step (int | float): Step size.
        dtype (Optional[Dtype]): Data type of the output tensor.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: 1-D tensor of evenly spaced values.

    Examples:
        >>> x = nova.arange(3)
        >>> print(x)
        tensor([0, 1, 2])

        >>> y = nova.arange(1, 5, 2)
        >>> print(y)
        tensor([1, 3])
    """
    if stop is None:
        stop = start
        start = 0

    if dtype is None:
        dtype = nova.long

    data = np.arange(start=start, stop=stop, step=step, dtype=dtype)

    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def unique(
    input: nova.Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: Dim = None,
):
    """
    Returns the unique elements of a tensor.

    Args:
        input (Tensor): Input tensor.
        sorted (bool): Whether to return sorted unique elements.
        return_inverse (bool): Whether to return indices to reconstruct original tensor.
        return_counts (bool): Whether to return counts of each unique element.
        dim (Dim): Dimension along which to find unique elements. Defaults to None.

    Returns:
        Tensor or tuple[Tensor, ...]: Tensor of unique elements, optionally with inverse indices and counts.

    Examples:
        >>> x = nova.tensor([1, 2, 2, 3])
        >>> nova.unique(x)
        tensor([1, 2, 3])

        >>> nova.unique(x, return_inverse=True, return_counts=True)
        tensor([1, 2, 3], [0, 1, 1, 2], [1, 2, 1])
    """
    input = ensure_tensor(input)

    unique = np.unique(
        input.data,
        sorted=sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=dim,
    )

    return nova.Tensor(unique, requires_grad=False, dtype=input.dtype)


def argmin(input: nova.Tensor, dim: Dim = None, keepdims: bool = False):
    """
    Returns the indices of the minimum values along a specified dimension.

    Args:
        input (Tensor): Input tensor.
        dim (Dim): Dimension along which to find minima. Defaults to None (flattened tensor).
        keepdims (bool): Whether to keep the reduced dimension.

    Returns:
        Tensor: Tensor of indices of minimum values.

    Examples:
        >>> x = nova.tensor([3, 1, 2])
        >>> nova.argmin(x)
        tensor(1)
    """
    input = ensure_tensor(input)

    return input.argmin(dim=dim, keepdims=keepdims)


def argmax(input: nova.Tensor, dim: Dim = None, keepdims: bool = False):
    """
    Returns the indices of the maximum values along a specified dimension.

    Args:
        input (Tensor): Input tensor.
        dim (Dim): Dimension along which to find maxima. Defaults to None (flattened tensor).
        keepdims (bool): Whether to keep the reduced dimension.

    Returns:
        Tensor: Tensor of indices of maximum values.

    Examples:
        >>> x = nova.tensor([3, 1, 2])
        >>> nova.argmax(x)
        tensor(0)
    """
    input = ensure_tensor(input)

    return input.argmax(dim=dim, keepdims=keepdims)


def argsort(input: nova.Tensor, dim: Dim = -1, kind=None, order=None):
    """
    Returns the indices that would sort a tensor along a given dimension.

    Args:
        input (Tensor): Input tensor.
        dim (Dim): Dimension along which to sort.
        kind (optional): Sorting algorithm.
        order (optional): Field order for structured arrays.

    Returns:
        Tensor: Indices that would sort the tensor.

    Examples:
        >>> x = nova.tensor([3, 1, 2])
        >>> nova.argsort(x)
        tensor([1, 2, 0])
    """
    input = ensure_tensor(input)

    return input.argsort(dim=dim, kind=kind, order=order)


def argwhere(input: nova.Tensor):
    """
    Returns the indices of non-zero elements in the input tensor.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Indices of non-zero elements.

    Examples:
        >>> x = nova.tensor([0, 1, 2])
        >>> nova.argwhere(x)
        tensor([[1], [2]])
    """
    input = ensure_tensor(input)

    return input.argwhere()


def stack(inputs: list[nova.Tensor], dim: Dim = 0):
    """
    Stacks a sequence of tensors along a new dimension.

    Args:
        inputs (list[Tensor]): List of tensors to stack.
        dim (Dim): Dimension along which to insert the new axis.

    Returns:
        Tensor: Stacked tensor.

    Examples:
        >>> a = nova.tensor([1, 2])
        >>> b = nova.tensor([3, 4])
        >>> nova.stack([a, b], dim=0)
        tensor([[1, 2],
                [3, 4]])
    """
    from nova.autograd._ops import Stack

    return Stack.apply(inputs, dim)


def zeros(
    size: Size, dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    """
    Returns a tensor filled with zeros.

    Args:
        size (Size): Shape of the tensor.
        dtype (Optional[Dtype]): Data type.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: Tensor of zeros.

    Examples:
        >>> x = nova.zeros((2,3))
        >>> print(x)
        tensor([[0, 0, 0],
                [ 0, 0, 0]])
    """
    dtype = dtype if dtype is not None else nova.float32
    data = np.zeros(size, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros_like(
    input: nova.Tensor, dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    """
    Returns a tensor of zeros with the same shape as input.

    Args:
        input (Tensor): Reference tensor.
        dtype (Optional[Dtype]): Data type of output tensor.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: Tensor of zeros matching `input` shape.

    Examples:
        >>> x = nova.tensor([[1,2],[3,4]])
        >>> nova.zeros_like(x)
        tensor([[0, 0],
                [0, 0]])
    """
    input = ensure_tensor(input)
    data = np.zeros_like(input.data, dtype=input.dtype if dtype is None else dtype)
    return nova.Tensor(
        data,
        requires_grad=requires_grad,
        dtype=input.dtype if dtype is None else dtype,
    )


def ones(
    size: Size, dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    """
    Returns a tensor filled with ones.

    Args:
        size (Size): Shape of the tensor.
        dtype (Optional[Dtype]): Data type.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: Tensor of ones.

    Examples:
        >>> nova.ones((2,2))
        tensor([[1,1],
                [1,1]])
    """
    dtype = dtype if dtype is not None else nova.float32
    data = np.ones(size, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def ones_like(
    input: nova.Tensor, dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    """
    Returns a tensor of ones with the same shape as input.

    Args:
        input (Tensor): Reference tensor.
        dtype (Optional[Dtype]): Data type of output tensor.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: Tensor of ones matching `input` shape.

    Examples:
        >>> x = nova.tensor([[0,0],[0,0]])
        >>> nova.ones_like(x)
        tensor([[1,1],
                [1,1]]
    """
    input = ensure_tensor(input)
    data = np.ones_like(input.data, dtype=input.dtype if dtype is None else dtype)
    return nova.Tensor(
        data,
        requires_grad=requires_grad,
        dtype=input.dtype if dtype is None else dtype,
    )


def as_strided(input: nova.Tensor, size: Size, strides: Size) -> nova.Tensor:
    """
    Returns a view of the input tensor with the given shape and strides.

    Args:
        input (Tensor): Input tensor.
        size (Size): Shape of the output tensor.
        strides (Size): Strides for each dimension.

    Returns:
        Tensor: Strided tensor view.

    Examples:
        >>> x = nova.tensor([1,2,3,4])
        >>> nova.as_strided(x, size=(2,2), strides=(1,1))
        tensor([[1,2],
               [2,3]])
    """
    input = ensure_tensor(input)

    return input.as_strided(size=size, strides=strides)


def linspace(
    start: int | float,
    stop: int | float,
    num: int | float,
    endpoint: Optional[bool] = True,
    dtype: Optional[Dtype] = None,
    dim: Optional[Dim] = 0,
    requires_grad: bool = False,
) -> nova.Tensor:
    """
    Returns a 1-D tensor of `num` evenly spaced points between `start` and `stop`.

    Args:
        start (int | float): Start value.
        stop (int | float): End value.
        num (int | float): Number of points.
        endpoint (bool): If true, stop is the last sample. Otherwise, it is not included.
        dtype (Optional[Dtype]): Data type.
        dim (Optional[Dim]):  The axis in the result to store the samples.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: Tensor of evenly spaced values.

    Examples:
        >>> nova.linspace(0, 1, 5)
        tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    """
    if dtype is None:
        dtype = nova.float32
    data = np.linspace(
        start=start, stop=stop, num=num, endpoint=endpoint, dtype=dtype, axis=dim
    )
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def logspace(
    start: int | float,
    stop: int | float,
    num: int | float,
    endpoint: Optional[bool] = True,
    dtype: Optional[Dtype] = None,
    dim: Optional[Dim] = 0,
    requires_grad: bool = False,
) -> nova.Tensor:
    """
    Returns a 1-D tensor of `num` evenly log spaced points between `start` and `stop`.

    Args:
        start (int | float): Start value.
        stop (int | float): End value.
        num (int | float): Number of points.
        endpoint (bool): If true, stop is the last sample. Otherwise, it is not included.
        dtype (Optional[Dtype]): Data type.
        dim (Optional[Dim]):  The axis in the result to store the samples.
        requires_grad (bool): Whether to track gradients.

    Returns:
        Tensor: Tensor of evenly log spaced values.

    Examples:
        >>> nova.logspace(1e-6, 1.0, 5)
        tensor([1.0, 1.78, 3.26, 5.62, 10.0])
    """
    if dtype is None:
        dtype = nova.float32
    data = np.logspace(
        start=start, stop=stop, num=num, endpoint=endpoint, dtype=dtype, axis=dim
    )
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def any(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.bool:
    """
    Returns True if any element is non-zero along a given dimension.

    Args:
        input (Tensor): Input tensor.
        dim (Dim): Dimension to reduce. Defaults to None.
        keepdims (bool): Whether to keep reduced dimensions.

    Returns:
        Tensor: Boolean tensor.

    Examples:
        >>> x = nova.tensor([[0,1],[0,0]])
        >>> nova.any(x, dim=0)
        [False, True]
    """
    input = ensure_tensor(input)
    return input.any(dim, keepdims=keepdims)


def all(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.bool:
    """
    Returns True if all elements are non-zero along a given dimension.

    Args:
        input (Tensor): Input tensor.
        dim (Dim): Dimension to reduce. Defaults to None.
        keepdims (bool): Whether to keep reduced dimensions.

    Returns:
        Tensor: Boolean tensor.

    Examples:
        >>> x = nova.tensor([[1,1],[0,1]])
        >>> nova.all(x, dim=1)
        [True, False]
    """
    input = ensure_tensor(input)
    return input.all(dim, keepdims=keepdims)


def allclose(
    input: nova.Tensor,
    other: nova.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> bool:
    """
    Returns True if all elements of two tensors are close within given tolerances.

    Args:
        input (Tensor): First tensor.
        other (Tensor): Second tensor.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        equal_nan (bool): Whether to compare NaNs as equal.

    Returns:
        bool: True if tensors are element-wise equal within tolerance.

    Examples:
        >>> x = nova.tensor([1.0, 2.0])
        >>> y = nova.tensor([1.0, 2.000001])
        >>> nova.allclose(x, y)
        True
    """
    input = ensure_tensor(input)
    other = ensure_tensor(other)

    return input.allclose(other=other, rtol=rtol, atol=atol, equal_nan=equal_nan)
