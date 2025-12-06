import pytest
import numpy as np
from novann.core import (
    kaiming_normal_,
    kaiming_uniform_,
    xavier_normal_,
    xavier_uniform_,
    calculate_gain,
    random_init_,
)

# Shapes para pruebas
shape2d = (200, 784)  # (out,in)
shape3d = (3, 128, 128)  # (out,in,l)
shape4d = (128, 3, 3, 3)  # (out,in,k,k)
shape5d = (128, 3, 3, 3, 3)  # (out,in,kd,k,k)

all_shapes = [shape2d, shape3d, shape4d, shape5d]
nonlinearity = "leakyrelu"


def test_kaiming_normal_distribution():
    """Check mean/variance behaviour of Kaiming normal initializer for all shapes."""
    for shape in all_shapes:
        W = kaiming_normal_(shape, nonlinearity=nonlinearity)

        from novann.core.init import shape_validation

        fan_in = shape_validation(shape, mode="fan_in")

        gain = calculate_gain(nonlinearity=nonlinearity)
        expected_std = gain / np.sqrt(fan_in)

        # Expect near-zero mean and variance in the rough neighborhood of expected_var
        assert abs(np.mean(W)) < 0.1, f"Mean test failed for shape {shape}"
        assert (
            abs(np.std(W) - expected_std) < 0.1
        ), f"Variance test failed for shape {shape}"


def test_kaiming_uniform_distribution():
    """Ensure values lie within the computed uniform bounds for all shapes."""
    for shape in all_shapes:
        W = kaiming_uniform_(shape, nonlinearity=nonlinearity)

        from novann.core.init import shape_validation

        fan_in = shape_validation(shape, mode="fan_in")
        gain = calculate_gain(nonlinearity=nonlinearity)

        limit = gain * np.sqrt(3.0 / fan_in)
        assert np.all(W <= limit) and np.all(
            W >= -limit
        ), f"Uniform bounds test failed for shape {shape}"


def test_xavier_normal_distribution():
    """Xavier normal should produce zero-mean and proper std scaling for all shapes."""
    for shape in all_shapes:
        gain = calculate_gain(nonlinearity=nonlinearity)
        W = xavier_normal_(shape, gain)

        from novann.core.init import shape_validation

        fan_in, fan_out = shape_validation(shape, mode="both")

        expected_var = gain * np.sqrt(2 / (fan_in + fan_out))

        assert abs(np.mean(W)) < 0.1, f"Mean test failed for shape {shape}"
        assert (
            abs(np.var(W) - expected_var) < 0.1
        ), f"Variance test failed for shape {shape}"


def test_xavier_uniform_distribution():
    """Xavier uniform should bound weights within computed limit for all shapes."""
    for shape in all_shapes:
        gain = calculate_gain(nonlinearity=nonlinearity)
        W = xavier_uniform_(shape, gain)

        from novann.core.init import shape_validation

        fan_in, fan_out = shape_validation(shape, mode="both")

        limit = gain * np.sqrt(6 / (fan_in + fan_out))

        assert np.all(W <= limit) and np.all(
            W >= -limit
        ), f"Uniform bounds test failed for shape {shape}"


def test_random_initializer():
    """Sanity check for the small random initializer (near-zero mean) for all shapes."""
    for shape in all_shapes:
        gain = calculate_gain(nonlinearity=nonlinearity)
        W = random_init_(shape, gain)
        assert abs(np.mean(W)) < 0.1, f"Random init mean test failed for shape {shape}"


def test_exceptions_of_init_methods():
    """Invalid nonlinearity names should raise ValueError for all shapes."""
    nonlinearity = "gelu"

    for shape in all_shapes:
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
