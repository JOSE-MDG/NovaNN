import pytest
import numpy as np
from src.core.init import (
    kaiming_normal_,
    kaiming_uniform_,
    xavier_normal_,
    xavier_uniform_,
    calculate_gain,
    random_init_,
)

shape = (200, 784)  # (out,in)
nonlinearity = "leakyrelu"


def test_kaiming_normal_distribution():

    W = kaiming_normal_(shape, nonlinearity=nonlinearity)

    fan_in = W.shape[1]

    gain = calculate_gain(nonlinearity=nonlinearity)
    expected_var = gain / np.sqrt(fan_in)

    assert abs(np.mean(W)) < 0.1
    assert abs(np.var(W) - expected_var) < 0.1


def test_kaiming_uniform_distribution():

    W = kaiming_uniform_(shape, nonlinearity=nonlinearity)
    fan_in = shape[1]
    gain = calculate_gain(nonlinearity=nonlinearity)

    limit = gain * np.sqrt(3.0 / fan_in)
    assert np.all(W <= limit) and np.all(W >= -limit)


def test_xavier_normal_distribution():

    W = xavier_normal_(shape, calculate_gain(nonlinearity=nonlinearity))
    fan_in = shape[1]
    fan_out = shape[0]
    gain = calculate_gain(nonlinearity=nonlinearity)

    expected_var = gain * np.sqrt(2 / (fan_in + fan_out))

    assert abs(np.mean(W)) < 0.1
    assert abs(np.var(W) - expected_var) < 0.1


def test_xavier_uniform_distribution():

    W = xavier_uniform_(shape, calculate_gain(nonlinearity=nonlinearity))
    fan_in = shape[1]
    fan_out = shape[0]
    gain = calculate_gain(nonlinearity=nonlinearity)

    limit = gain * np.sqrt(6 / (fan_in + fan_out))

    assert np.all(W <= limit) and np.all(W >= -limit)


def test_random_initializer():

    W = random_init_(shape, calculate_gain(nonlinearity=nonlinearity))
    assert abs(np.mean(W)) < 0.1


def test_exceptions_of_init_methos():

    nonlinearity = "gelu"
    with pytest.raises(ValueError):
        W = kaiming_normal_(shape, nonlinearity=nonlinearity)

    with pytest.raises(ValueError):
        W = kaiming_uniform_(shape, nonlinearity=nonlinearity)

    with pytest.raises(ValueError):
        gain = calculate_gain(nonlinearity=nonlinearity)
        W = xavier_normal_(shape, gain)

    with pytest.raises(ValueError):
        gain = calculate_gain(nonlinearity=nonlinearity)
        W = xavier_uniform_(shape, gain)
