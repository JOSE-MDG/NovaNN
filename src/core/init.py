import numpy as np


def calculate_gain(nonlinearity: str, param: float = None):

    if nonlinearity == "linear" or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3.0
    elif nonlinearity == "relu":
        return np.sqrt(2.0)
    elif nonlinearity == "leakyrelu":
        if param is None:
            negative_slope = 0.01
        else:
            negative_slope = param
        return np.sqrt(2.0 / (1 + negative_slope**2))
    else:
        raise ValueError(f"Unsoported activation function: {nonlinearity}")


def xavier_normal_(shape: tuple[int, int], gain: float = 1.0) -> np.ndarray:
    if len(shape) < 2:
        raise ValueError(f"The shape must be grater than {len(shape)}")

    fan_in = shape[1]  # inputs
    fan_out = shape[0]  # outputs

    std = gain * np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0.0, std, shape)


def xavier_uniform_(shape: tuple[int, int], gain: float = 1.0) -> np.ndarray:
    if len(shape) < 2:
        raise ValueError(f"The shape must be grater than {len(shape)}")

    fan_in = shape[1]
    fan_out = shape[0]

    limit = gain * np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)


def kaiming_normal_(
    shape: tuple[int, int], a: float, nonlinearity: str = "relu", mode: str = "fan_in"
) -> np.ndarray:
    if len(shape) < 2:
        raise ValueError(f"The shape must be grater than {len(shape)}")

    fan_in = shape[1]
    fan_out = shape[0]

    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError(f"mode must be 'fan_in' or 'fan_out not {mode}")

    gain = calculate_gain(nonlinearity=nonlinearity, param=a)
    std = gain / np.sqrt(fan)
    return np.random.normal(0.0, std, shape)


def kaiming_uniform_(
    shape: tuple[int, int], a: float, nonlinearity: str = "relu", mode: str = "fan_in"
) -> np.ndarray:
    if len(shape) < 2:
        raise ValueError(f"The shape must be grater than {len(shape)}")

    fan_in = shape[1]
    fan_out = shape[0]

    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError(f"mode must be 'fan_in' or 'fan_out not {mode}")

    gain = calculate_gain(nonlinearity=nonlinearity, param=a)
    limit = gain * np.sqrt(3.0 / fan)
    return np.random.uniform(-limit, limit, shape)


def random_init_(shape: tuple[int, int], gain: float = 0.001) -> np.ndarray:
    return np.random.randn(shape) * gain
