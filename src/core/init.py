import numpy as np
from typing import Optional, Union
from src._typing import Shape

"""
Weight initialization utilities.

Provides common initializers (Xavier/Glorot, Kaiming/He) and a small random 
initializer used as default. All functions use Google-style docstrings.
"""


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """Return the recommended gain value for the given nonlinearity.

    Args:
        nonlinearity: Name of the activation function. Supported values:
            "linear", "sigmoid", "tanh", "relu", "leakyrelu".
        param: Optional parameter for some nonlinearities (e.g., negative
            slope for leaky ReLU). Uses sensible defaults if None.

    Returns:
        Gain multiplier as float.

    Raises:
        ValueError: If `nonlinearity` is not supported.
    """
    if nonlinearity in ("linear", "sigmoid"):
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3.0
    elif nonlinearity == "relu":
        return float(np.sqrt(2.0))
    elif nonlinearity == "leakyrelu":
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
    valid_modes = ("both", "fan_in", "fan_out")
    if mode not in valid_modes:
        raise ValueError(f"Mode must be {valid_modes}, got '{mode}'")


def _calculate_fans(shape: Shape) -> tuple[int, int]:
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


def shape_validation(shape: Shape, mode: str = "fan_in") -> Union[int, tuple[int, int]]:
    """Calculate fan values for weight initialization.

    Args:
        shape: Weight tensor shape.
        mode: One of 'fan_in', 'fan_out', or 'both'.

    Returns:
        Single fan value or tuple (fan_in, fan_out) depending on mode.

    Raises:
        ValueError: If mode is invalid or shape has invalid dimensions.
    """
    _validate_mode(mode)
    fan_in, fan_out = _calculate_fans(shape)

    if mode == "both":
        return fan_in, fan_out
    elif mode == "fan_in":
        return fan_in
    else:  # mode == "fan_out"
        return fan_out


def xavier_normal_(shape: Shape, gain: float = 1.0) -> np.ndarray:
    """Xavier (Glorot) normal initialization.

    Args:
        shape: Weight tensor shape (fan_out, fan_in, ...).
        gain: Optional scaling factor.

    Returns:
        Array sampled from N(0, std^2) with given shape.

    Raises:
        ValueError: If shape has fewer than 2 dimensions.
    """
    fan_in, fan_out = shape_validation(shape=shape, mode="both")

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0.0, std, shape).astype(np.float32)


def xavier_uniform_(shape: Shape, gain: float = 1.0) -> np.ndarray:
    """Xavier (Glorot) uniform initialization.

    Args:
        shape: Weight tensor shape (fan_out, fan_in, ...).
        gain: Optional scaling factor.

    Returns:
        Array sampled uniformly in [-limit, limit] with given shape.

    Raises:
        ValueError: If shape has fewer than 2 dimensions.
    """
    fan_in, fan_out = shape_validation(shape=shape, mode="both")

    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape).astype(np.float32)


def kaiming_normal_(
    shape: Shape,
    a: Optional[float] = None,
    nonlinearity: str = "relu",
    mode: str = "fan_in",
) -> np.ndarray:
    """Kaiming (He) normal initialization.

    Args:
        shape: Weight tensor shape.
        a: Negative slope for leaky ReLU (used to compute gain).
        nonlinearity: Activation name used to compute gain.
        mode: Either 'fan_in' or 'fan_out' to control scaling.

    Returns:
        Array sampled from N(0, std^2) with given shape.

    Raises:
        ValueError: If shape has invalid dimensions or mode is invalid.
    """
    fan = shape_validation(shape=shape, mode=mode)
    gain = calculate_gain(nonlinearity=nonlinearity, param=a)

    std = gain / np.sqrt(fan)
    return np.random.normal(0.0, std, shape).astype(np.float32)


def kaiming_uniform_(
    shape: Shape,
    a: Optional[float] = None,
    nonlinearity: str = "relu",
    mode: str = "fan_in",
) -> np.ndarray:
    """Kaiming (He) uniform initialization.

    Args:
        shape: Weight tensor shape.
        a: Negative slope for leaky ReLU (used to compute gain).
        nonlinearity: Activation name used to compute gain.
        mode: Either 'fan_in' or 'fan_out' to control scaling.

    Returns:
        Array sampled uniformly in [-limit, limit] with given shape.

    Raises:
        ValueError: If shape has invalid dimensions or mode is invalid.
    """
    fan = shape_validation(shape=shape, mode=mode)
    gain = calculate_gain(nonlinearity=nonlinearity, param=a)

    limit = gain * np.sqrt(3.0 / fan)
    return np.random.uniform(-limit, limit, shape).astype(np.float32)


def random_init_(shape: Shape, gain: float = 0.001) -> np.ndarray:
    """Small random normal initializer (conservative default).

    Args:
        shape: Output tensor shape.
        gain: Scale multiplier applied to standard normal samples.

    Returns:
        Array sampled from N(0, 1) scaled by `gain`.
    """
    return np.random.randn(*shape).astype(np.float32) * gain
