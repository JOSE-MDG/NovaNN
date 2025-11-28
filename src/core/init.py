import numpy as np
from typing import Optional, Tuple

"""
Weight initialization utilities.

Functions follow the common initializers (Xavier / Glorot, Kaiming / He) and a
small random initializer used as a default. Docstrings use the Google style.
"""

Shape2D = Tuple[int, int]


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """Return the recommended gain value for the given nonlinearity.

    Args:
        nonlinearity: Name of the activation function. Supported values include
            "linear", "sigmoid", "tanh", "relu", "leakyrelu".
        param: Optional parameter used by some nonlinearities (e.g. negative
            slope for leaky ReLU). If None, sensible defaults are used.

    Returns:
        The gain multiplier as a float.

    Raises:
        ValueError: If `nonlinearity` is not recognised.
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


def xavier_normal_(shape: Shape2D, gain: float = 1.0) -> np.ndarray:
    """Xavier (Glorot) normal initialization.

    Args:
        shape: Tuple (fan_out, fan_in).
        gain: Optional scaling factor (often derived from activation gain).

    Returns:
        Array sampled from N(0, std^2) with the given shape.

    Raises:
        ValueError: If `shape` has fewer than 2 dimensions.
    """
    if len(shape) < 2:
        raise ValueError(f"The shape must be at least 2D, got {len(shape)}")

    fan_in = shape[1]
    fan_out = shape[0]

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0.0, std, shape)


def xavier_uniform_(shape: Shape2D, gain: float = 1.0) -> np.ndarray:
    """Xavier (Glorot) uniform initialization.

    Args:
        shape: Tuple (fan_out, fan_in).
        gain: Optional scaling factor.

    Returns:
        Array sampled uniformly in [-limit, limit] with the given shape.

    Raises:
        ValueError: If `shape` has fewer than 2 dimensions.
    """
    if len(shape) < 2:
        raise ValueError(f"The shape must be at least 2D, got {len(shape)}")

    fan_in = shape[1]
    fan_out = shape[0]

    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)


def kaiming_normal_(
    shape: Shape2D,
    a: Optional[float] = None,
    nonlinearity: str = "relu",
    mode: str = "fan_in",
) -> np.ndarray:
    """Kaiming (He) normal initialization.

    Args:
        shape: Tuple (fan_out, fan_in).
        a: Optional negative slope for leaky ReLU; used to compute gain.
        nonlinearity: Activation name used to compute gain.
        mode: Either "fan_in" or "fan_out" to control scaling.

    Returns:
        Array sampled from N(0, std^2) with the given shape.

    Raises:
        ValueError: If `shape` has fewer than 2 dimensions or `mode` is invalid.
    """
    if len(shape) < 2:
        raise ValueError(f"The shape must be at least 2D, got {len(shape)}")

    fan_in = shape[1]
    fan_out = shape[0]

    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', not {mode}")

    gain = calculate_gain(nonlinearity=nonlinearity, param=a)
    std = gain / np.sqrt(fan)
    return np.random.normal(0.0, std, shape)


def kaiming_uniform_(
    shape: Shape2D,
    a: Optional[float] = None,
    nonlinearity: str = "relu",
    mode: str = "fan_in",
) -> np.ndarray:
    """Kaiming (He) uniform initialization.

    Args:
        shape: Tuple (fan_out, fan_in).
        a: Optional negative slope for leaky ReLU; used to compute gain.
        nonlinearity: Activation name used to compute gain.
        mode: Either "fan_in" or "fan_out" to control scaling.

    Returns:
        Array sampled uniformly in [-limit, limit] with the given shape.

    Raises:
        ValueError: If `shape` has fewer than 2 dimensions or `mode` is invalid.
    """
    if len(shape) < 2:
        raise ValueError(f"The shape must be at least 2D, got {len(shape)}")

    fan_in = shape[1]
    fan_out = shape[0]

    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', not {mode}")

    gain = calculate_gain(nonlinearity=nonlinearity, param=a)
    limit = gain * np.sqrt(3.0 / fan)
    return np.random.uniform(-limit, limit, shape)


def random_init_(shape: Shape2D, gain: float = 0.001) -> np.ndarray:
    """Small random normal initializer (used as a conservative default).

    Args:
        shape: Tuple (fan_out, fan_in).
        gain: Scale multiplier applied to standard normal samples.

    Returns:
        Array sampled from N(0, 1) scaled by `gain`.
    """
    return np.random.randn(*shape) * gain
