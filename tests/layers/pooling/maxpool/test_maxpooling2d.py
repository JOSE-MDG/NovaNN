import numpy as np
import pytest

from novann.layers import MaxPool2d
from novann.utils.gradient_checking import numeric_grad_scalar_wrt_x


RNG = np.random.RandomState(8)


def test_maxpool2d_forward_shape():
    """Test forward pass output shape."""
    layer = MaxPool2d(kernel_size=2, stride=2, padding=0)
    x = RNG.randn(4, 3, 10, 10).astype(np.float32)
    output = layer(x)

    # H_out = floor((10 + 0 - 2) / 2) + 1 = 5
    assert output.shape == (4, 3, 5, 5), f"Expected (4, 3, 5, 5), got {output.shape}"


def test_maxpool2d_forward_padding():
    """Test forward pass with padding."""
    layer = MaxPool2d(kernel_size=3, stride=1, padding=1)
    x = RNG.randn(2, 4, 5, 5).astype(np.float32)
    output = layer(x)

    # H_out = floor((5 + 2*1 - 3) / 1) + 1 = 5
    assert output.shape == (2, 4, 5, 5), f"Expected (2, 4, 5, 5), got {output.shape}"


def test_maxpool2d_backward_gradient():
    """Gradient checking for MaxPool2d input gradient."""
    layer = MaxPool2d(kernel_size=2, stride=2, padding=0)

    x = RNG.randn(2, 3, 6, 6).astype(np.float32) * 0.1
    output = layer(x)
    G = RNG.randn(*output.shape).astype(np.float32) * 0.1

    analytic_grad = layer.backward(G)

    # Numerical gradient
    layer_copy = MaxPool2d(kernel_size=2, stride=2, padding=0)
    numeric_grad = numeric_grad_scalar_wrt_x(
        lambda x_input: layer_copy(x_input), x, G, eps=1e-5
    )

    # set a comparison threshold
    THRESHOLD = 5e-3

    diff = np.abs(analytic_grad - numeric_grad).max()
    assert diff < THRESHOLD, f"Input gradient mismatch: {diff:.2e}"
