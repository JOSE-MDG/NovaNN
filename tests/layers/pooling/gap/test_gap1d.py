import numpy as np
import pytest

from novann.layers import GlobalAvgPool1d
from novann.utils.gradient_checking import numeric_grad_scalar_wrt_x

RNG = np.random.RandomState(8)


def test_global_avg_pool1d_forward_shape():
    """Test forward pass output shape."""
    layer = GlobalAvgPool1d()
    x = RNG.randn(4, 3, 10).astype(np.float32)
    output = layer(x)

    assert output.shape == (4, 3, 1), f"Expected (4, 3, 1), got {output.shape}"


def test_global_avg_pool1d_forward_values():
    """Test forward pass numerical values."""
    layer = GlobalAvgPool1d()

    # Test with constant values
    x = np.ones((2, 3, 5), dtype=np.float32) * 2.0
    output = layer(x)

    assert np.allclose(output, 2.0), f"Expected all 2.0, got {output}"


def test_global_avg_pool1d_backward_gradient():
    """Gradient checking for GlobalAvgPool1d."""
    layer = GlobalAvgPool1d()

    x = RNG.randn(2, 3, 6).astype(np.float32) * 0.1
    output = layer(x)
    G = RNG.randn(*output.shape).astype(np.float32) * 0.1

    analytic_grad = layer.backward(G)

    # Numerical gradient
    layer_copy = GlobalAvgPool1d()
    numeric_grad = numeric_grad_scalar_wrt_x(
        lambda x_input: layer_copy(x_input), x, G, eps=1e-5
    )

    # Set a threshold
    THRESHOLD = 5e-3

    diff = np.abs(analytic_grad - numeric_grad).max()
    assert diff < THRESHOLD, f"Input gradient mismatch: {diff:.2e}"


def test_global_avg_pool1d_uniform_gradient():
    """Test that gradient is uniformly distributed."""
    layer = GlobalAvgPool1d()

    x = RNG.randn(1, 2, 10).astype(np.float32)
    layer(x)

    # Gradient of ones
    G = np.ones((1, 2, 1), dtype=np.float32)
    grad = layer.backward(G)

    # Each input element should receive gradient = 1/10 = 0.1
    expected = 1.0 / 10
    assert np.allclose(
        grad, expected
    ), f"Expected uniform gradient {expected}, got {grad[0,0,0]:.6f}"
