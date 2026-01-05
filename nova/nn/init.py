from __future__ import annotations
import numpy as np
from typing import Any, Literal, Optional, Union, TYPE_CHECKING


if TYPE_CHECKING:
    from nova.nn import Parameter, Buffer
    from nova import Tensor
    from nova._typing import Size

"""
Parameter initialization utilities.

This module provides common weight initialization methods used in neural
networks, including Xavier/Glorot and Kaiming/He initializations, as well as
simple uniform, normal, and constant initializers.

The functions operate *in-place* on Tensors, Parameters, or Buffers and
temporarily disable gradient tracking during initialization.

Design notes:
- All initializers follow conventions used in modern deep learning frameworks
  (e.g., PyTorch).
- Fan-in and fan-out are inferred automatically from tensor shapes.
- Boolean tensors are not supported as initialization targets.

Typical usage:
    >>> import nova
    >>> form nova.nn import init
    >>> w = Parameter(nova.empty((128, 256)), dtype=nova.float32)
    >>> init.kaiming_normal_(w, nonlinearity="relu")

Available initializers:
- Xavier / Glorot: xavier_normal_, xavier_uniform_
- Kaiming / He: kaiming_normal_, kaiming_uniform_
- Basic: uniform_, normal_, constant_, zeros_, ones_, random_
"""


def calculate_gain(
    nonlinearity: Literal[
        "linear", "sigmoid", "tanh", "relu", "leaky_relu", "prelu", "gelu"
    ],
    param: Optional[float] = None,
) -> float:
    """
    Return the recommended gain value for a given nonlinearity.

    The gain is used to scale the variance of weight initializations in
    Xavier/Glorot and Kaiming/He methods.

    Args:
        nonlinearity: Name of the activation function. Supported values:
            "linear", "sigmoid", "tanh", "relu", "leaky_relu", "prelu", "gelu".
        param: Optional parameter for some nonlinearities (e.g., negative
            slope for leaky ReLU). Uses sensible defaults if None.

    Returns:
        Gain multiplier as a float.

    Examples:
        >>> calculate_gain("relu")
        1.4142135623730951
        >>> calculate_gain("leaky_relu", param=0.2)
        1.3867504905630728

    Raises:
        ValueError: If `nonlinearity` is not supported.
    """
    if nonlinearity in ("linear", "sigmoid"):
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3.0
    elif nonlinearity == "relu" or nonlinearity == "gelu":
        return float(np.sqrt(2.0))
    elif nonlinearity == "leaky_relu" or nonlinearity == "prelu":
        negative_slope = 0.01 if param is None else float(param)
        return float(np.sqrt(2.0 / (1 + negative_slope**2)))
    else:
        raise ValueError(f"Unsupported activation function: {nonlinearity}")


def _validate_mode(mode: str) -> None:
    """Validate initialization mode.

    Args:
        mode: Mode to validate.

    Raises:
        ValueError: If mode is not 'both', 'fan_in', or 'fan_out'.
    """
    valid_modes: Literal["both", "fan_in", "fan_out"] = ("both", "fan_in", "fan_out")
    if mode not in valid_modes:
        raise ValueError(f"Mode must be {valid_modes}, got '{mode}'")


def _calculate_fans(shape: Size) -> tuple[int, int]:
    """Calculate fan_in and fan_out from shape.

    Args:
        shape: Weight shape of 2 to 5 dimensions.

    Returns:
        Tuple of (fan_in, fan_out).

    Raises:
        ValueError: If shape has invalid number of dimensions.
    """
    if len(shape) < 2:
        raise ValueError(f"Shape must have at least 2 dimensions, got {len(shape)}")

    # Linear layers: (out_features, in_features)
    if len(shape) == 2:
        fan_out, fan_in = shape
        receptive_field_size = 1
    # 1D layers: (out_channels, in_channels, kernel_size)
    elif len(shape) == 3:
        fan_out, fan_in, receptive_field_size = shape
    # 2D layers: (out_channels, in_channels, kernel_height, kernel_width)
    elif len(shape) == 4:
        fan_out, fan_in, kh, kw = shape
        receptive_field_size = kh * kw
    # 3D layers: (out_channels, in_channels, kd, kh, kw)
    elif len(shape) == 5:
        fan_out, fan_in, kd, kh, kw = shape
        receptive_field_size = kd * kh * kw
    else:
        raise ValueError(f"Shape must have 2 to 5 dimensions, got {len(shape)}")

    fan_in *= receptive_field_size
    fan_out *= receptive_field_size
    return fan_in, fan_out


def get_fans(
    tensor: Tensor, mode: Literal["both", "fan_in", "fan_out"] = "fan_in"
) -> Union[int, tuple[int, int]]:
    """
    Compute fan-in and fan-out values for a weight tensor.

    Fan values are used to scale the variance of initialization distributions.
    The calculation depends on the tensor shape and supports linear and convolutional layers.

    Args:
        tensor: Tensor whose shape is used to compute fan values.
        mode: One of:
            - "fan_in": return only fan-in
            - "fan_out": return only fan-out
            - "both": return (fan_in, fan_out)

    Returns:
        An integer fan value or a tuple (fan_in, fan_out) if mode="both".

    Examples:
        >>> import nova
        >>> w = nova.randn(64, 128)
        >>> get_fans(w, mode="both")
        (128, 64)

    Raises:
        ValueError: If mode is invalid or tensor shape is unsupported.
    """

    shape = tensor.shape
    _validate_mode(mode)
    fan_in, fan_out = _calculate_fans(shape)

    if mode == "both":
        return fan_in, fan_out
    elif mode == "fan_in":
        return fan_in
    else:  # mode == "fan_out"
        return fan_out


def xavier_normal_(tensor: Parameter | Buffer, gain: float = 1.0) -> None:
    """
    Initialize tensor using Xavier (Glorot) normal initialization.

    The tensor is filled with values drawn from:
        N(0, gain * sqrt(2 / (fan_in + fan_out)))

    Args:
        tensor: Parameter or Buffer to initialize.
        gain: Optional scaling factor (see `calculate_gain`).

    Example:
        >>> import nova
        >>> from nova.nn import init, Parameter
        >>> w = Parameter(nova.empty((128, 256)), dtype=nova.float32)
        >>> init.xavier_normal_(w)
    """

    fan_in, fan_out = get_fans(tensor, mode="both")

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))

    prev_state = tensor.requires_grad

    tensor.requires_grad_(False)
    tensor.normal_(0.0, std)
    tensor.requires_grad_(prev_state)


def xavier_uniform_(tensor: Parameter | Buffer, gain: float = 1.0) -> None:
    """
    Initialize tensor using Xavier (Glorot) uniform initialization.

    Values are drawn from:
        U(-limit, limit), where limit = gain * sqrt(6 / (fan_in + fan_out))

    Args:
        tensor: Parameter or Buffer to initialize.
        gain: Optional scaling factor.

    Example:
        >>> import nova
        >>> from nova.nn import init, Parameter
        >>> w = Parameter(nova.empty((128, 256)), dtype=nova.float32)
        >>> init.xavier_uniform_(w)
    """

    fan_in, fan_out = get_fans(tensor, mode="both")

    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))

    prev_state = tensor.requires_grad

    tensor.requires_grad_(False)
    tensor.uniform_(-limit, limit)
    tensor.requires_grad_(prev_state)


def kaiming_normal_(
    tensor: Parameter | Buffer,
    a: Optional[float] = None,
    nonlinearity: str = "leaky_relu",
    mode: str = "fan_in",
) -> None:
    """
    Initialize tensor using Kaiming (He) normal initialization.

    Commonly used for ReLU-like nonlinearities.

    Args:
        tensor: Parameter or Buffer to initialize.
        a: Optional negative slope for leaky ReLU.
        nonlinearity: Activation function name.
        mode: One of 'fan_in' or 'fan_out'.

    Example:
        >>> import nova
        >>> from nova.nn import init, Parameter
        >>> w = Parameter(nova.empty((64, 128)), dtype=nova.float32)
        >>> init.kaiming_normal_(w, nonlinearity="relu")
    """
    fan = get_fans(tensor, mode=mode)
    gain = calculate_gain(nonlinearity=nonlinearity, param=a)

    std = gain / np.sqrt(fan)

    prev_state = tensor.requires_grad

    tensor.requires_grad_(False)
    tensor.normal_(0.0, std)
    tensor.requires_grad_(prev_state)


def kaiming_uniform_(
    tensor: Parameter | Buffer,
    a: Optional[float] = None,
    nonlinearity: str = "relu",
    mode: str = "fan_in",
) -> None:
    """
    Initialize tensor using Kaiming (He) uniform initialization.

    Args:
        tensor: Parameter or Buffer to initialize.
        a: Optional negative slope for leaky ReLU.
        nonlinearity: Activation function name.
        mode: One of 'fan_in' or 'fan_out'.

    Example:
        >>> import nova
        >>> from nova.nn import init, Parameter
        >>> w = Parameter(nova.empty((64, 128)), dtype=nova.float32)
        >>> init.kaiming_uniform_(w)
    """

    fan = get_fans(tensor, mode=mode)
    gain = calculate_gain(nonlinearity=nonlinearity, param=a)

    limit = gain * np.sqrt(3.0 / fan)

    prev_state = tensor.requires_grad

    tensor.requires_grad_(False)
    tensor.uniform_(-limit, limit)
    tensor.requires_grad_(prev_state)


def uniform_(
    tensor: Tensor | Parameter | Buffer, low: float = 0, high: float = 1
) -> None:
    """Fill tensor with values drawn from a uniform distribution."""

    prev_state = tensor.requires_grad

    tensor.requires_grad_(False)
    tensor.uniform_(low, high)
    tensor.requires_grad_(prev_state)


def normal_(
    tensor: Tensor | Parameter | Buffer, mean: float = 0, std: float = 1
) -> None:
    """Fill tensor with values drawn from a normal distribution."""

    prev_state = tensor.requires_grad

    tensor.requires_grad_(False)
    tensor.normal_(mean, std)
    tensor.requires_grad_(prev_state)


def constant_(tensor: Tensor | Parameter | Buffer, val: Any) -> None:
    """Fill tensor with a constant value."""

    prev_state = tensor.requires_grad

    tensor.requires_grad_(False)
    tensor.fill_(val)
    tensor.requires_grad_(prev_state)


def zeros_(tensor: Tensor | Parameter | Buffer) -> None:
    """Fill tensor with zeros."""

    prev_state = tensor.requires_grad

    tensor.requires_grad_(False)
    tensor.zero_()
    tensor.requires_grad_(prev_state)


def ones_(tensor: Tensor | Parameter | Buffer) -> None:
    """Fill tensor with ones."""

    prev_state = tensor.requires_grad

    tensor.requires_grad_(False)
    tensor.ones_()
    tensor.requires_grad_(prev_state)


def random_(tensor: Tensor | Parameter | Buffer) -> None:
    """Fill tensor with random values using the tensor's default RNG."""

    prev_state = tensor.requires_grad

    tensor.requires_grad_(False)
    tensor.random_()
    tensor.requires_grad_(prev_state)
