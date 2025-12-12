import numpy as np
import novann as nn
import pytest

from novann.utils.gradient_checking import numeric_grad_scalar_wrt_x

RNG = np.random.RandomState(8)


def test_global_avg_pool2d_forward_shape():
    """Test forward pass output shape."""
    layer = nn.GlobalAvgPool2d()
    x = RNG.randn(4, 3, 10, 12).astype(np.float32)
    output = layer(x)

    assert output.shape == (4, 3, 1, 1), f"Expected (4, 3, 1, 1), got {output.shape}"


def test_global_avg_pool2d_forward_values():
    """Test forward pass numerical values."""
    layer = nn.GlobalAvgPool2d()

    # Test with constant values
    x = np.ones((2, 3, 4, 5), dtype=np.float32) * 3.0
    output = layer(x)

    assert np.allclose(output, 3.0), f"Expected all 3.0, got {output}"


def test_global_avg_pool2d_backward_gradient():
    """Gradient checking for GlobalAvgPool2d."""
    layer = nn.GlobalAvgPool2d()

    x = RNG.randn(2, 3, 6, 6).astype(np.float32) * 0.1
    output = layer(x)
    G = RNG.randn(*output.shape).astype(np.float32) * 0.1

    analytic_grad = layer.backward(G)

    # Numerical gradient
    layer_copy = nn.GlobalAvgPool2d()

    numeric_grad = numeric_grad_scalar_wrt_x(
        lambda x_input: layer_copy(x_input), x, G, eps=1e-5
    )

    # set a comprison threshold
    THRESHOLD = 5e-3

    diff = np.abs(analytic_grad - numeric_grad).max()
    assert diff < THRESHOLD, f"Input gradient mismatch: {diff:.2e}"


def test_global_avg_pool2d_uniform_gradient():
    """Test that gradient is uniformly distributed."""
    layer = nn.GlobalAvgPool2d()

    x = RNG.randn(1, 2, 4, 5).astype(np.float32)
    layer(x)

    # Gradient of ones
    G = np.ones((1, 2, 1, 1), dtype=np.float32)
    grad = layer.backward(G)

    # Each input element should receive gradient = 1/(4*5) = 0.05
    expected = 1.0 / (4 * 5)
    assert np.allclose(
        grad, expected
    ), f"Expected uniform gradient {expected}, got {grad[0,0,0,0]:.6f}"
