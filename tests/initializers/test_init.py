import pytest
import numpy as np
from novann.core.init import (
    kaiming_normal_,
    kaiming_uniform_,
    xavier_normal_,
    xavier_uniform_,
    calculate_gain,
    random_init_,
)

# Typical dense layer shape: (out_features, in_features)
shape = (200, 784)  # (out,in)
nonlinearity = "leakyrelu"


def test_kaiming_normal_distribution():
    """Check mean/variance behaviour of Kaiming normal initializer."""
    W = kaiming_normal_(shape, nonlinearity=nonlinearity)

    fan_in = W.shape[1]
    gain = calculate_gain(nonlinearity=nonlinearity)
    expected_var = gain / np.sqrt(fan_in)

    # Expect near-zero mean and variance in the rough neighborhood of expected_var
    assert abs(np.mean(W)) < 0.1
    assert abs(np.var(W) - expected_var) < 0.1


def test_kaiming_uniform_distribution():
    """Ensure values lie within the computed uniform bounds."""
    W = kaiming_uniform_(shape, nonlinearity=nonlinearity)
    fan_in = shape[1]
    gain = calculate_gain(nonlinearity=nonlinearity)

    limit = gain * np.sqrt(3.0 / fan_in)
    assert np.all(W <= limit) and np.all(W >= -limit)


def test_xavier_normal_distribution():
    """Xavier normal should produce zero-mean and proper std scaling."""
    W = xavier_normal_(shape, calculate_gain(nonlinearity=nonlinearity))
    fan_in = shape[1]
    fan_out = shape[0]
    gain = calculate_gain(nonlinearity=nonlinearity)

    expected_var = gain * np.sqrt(2 / (fan_in + fan_out))

    assert abs(np.mean(W)) < 0.1
    assert abs(np.var(W) - expected_var) < 0.1


def test_xavier_uniform_distribution():
    """Xavier uniform should bound weights within computed limit."""
    W = xavier_uniform_(shape, calculate_gain(nonlinearity=nonlinearity))
    fan_in = shape[1]
    fan_out = shape[0]
    gain = calculate_gain(nonlinearity=nonlinearity)

    limit = gain * np.sqrt(6 / (fan_in + fan_out))

    assert np.all(W <= limit) and np.all(W >= -limit)


def test_random_initializer():
    """Sanity check for the small random initializer (near-zero mean)."""
    W = random_init_(shape, calculate_gain(nonlinearity=nonlinearity))
    assert abs(np.mean(W)) < 0.1


def test_exceptions_of_init_methos():
    """Invalid nonlinearity names should raise ValueError from calculate_gain / initializers."""
    nonlinearity = "gelu"
    with pytest.raises(ValueError):
        _ = kaiming_normal_(shape, nonlinearity=nonlinearity)

    with pytest.raises(ValueError):
        _ = kaiming_uniform_(shape, nonlinearity=nonlinearity)

    with pytest.raises(ValueError):
        gain = calculate_gain(nonlinearity=nonlinearity)
        _ = xavier_normal_(shape, gain)

    with pytest.raises(ValueError):
        gain = calculate_gain(nonlinearity=nonlinearity)
        _ = xavier_uniform_(shape, gain)
