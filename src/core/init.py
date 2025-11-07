import numpy as np


def xavier_normal_(shape: tuple[int, int]) -> np.ndarray:
    fan_in = shape[1]
    fan_out = shape[0]
    inits = np.random.randn(*shape) / (2 / (fan_in + fan_out))
    return inits


def xavier_uniform_(shape: tuple[int, int]) -> np.ndarray:
    fan_in = shape[1]
    fan_out = shape[0]
    inits = np.random.randn(*shape) / (6 / (fan_in + fan_out))
    return inits


def kaiming_normal_(shape: tuple[int, int]) -> np.ndarray:
    fan_in = shape[1]
    inits = np.random.randn(*shape) / (2 / fan_in)
    return inits


def kaiming_uniform_(shape: tuple[int, int]) -> np.ndarray:
    fan_in = shape[1]
    inits = np.random.randn(*shape) / (2 / fan_in)
    return inits


def random_init_(shape: tuple[int, int]) -> np.ndarray:
    return np.random.randn(*shape) * 0.001
