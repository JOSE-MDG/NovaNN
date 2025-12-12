# tests/test_layers_minimal.py
import numpy as np
import pytest
import novann as nn
from novann.utils.gradient_checking import numeric_grad_wrt_param


RNG = np.random.RandomState(8)


def test_linear_forward_shape():
    """Test forward pass output shape."""
    layer = nn.Linear(in_features=10, out_features=5, bias=True)
    x = RNG.randn(32, 10).astype(np.float32)  # batch_size=32
    output = layer(x)

    assert output.shape == (32, 5), f"Expected (32, 5), got {output.shape}"
    assert output.dtype == np.float32, f"Expected float32, got {output.dtype}"


def test_linear_forward_no_bias():
    """Test forward pass without bias."""
    layer = nn.Linear(in_features=10, out_features=5, bias=False)
    x = RNG.randn(32, 10).astype(np.float32)
    output = layer(x)

    assert layer.bias is None, "Bias should be None when bias=False"
    assert output.shape == (32, 5)


def test_linear_backward_gradient_check():
    """Gradient checking for Linear layer parameters."""
    layer = nn.Linear(in_features=4, out_features=3, bias=True)

    # Random input and gradient
    x = RNG.randn(2, 4).astype(np.float32) * 0.1
    G = RNG.randn(2, 3).astype(np.float32) * 0.1

    # Forward pass
    output = layer(x)

    # Backward pass (analytic gradients)
    grad_input = layer.backward(G)

    # Numerical gradient for weight
    num_grad_weight = numeric_grad_wrt_param(layer, "weight", x, G, eps=1e-5)

    # Numerical gradient for bias
    num_grad_bias = numeric_grad_wrt_param(layer, "bias", x, G, eps=1e-5)

    # Compare analytic vs numerical gradients
    weight_diff = np.abs(layer.weight.grad - num_grad_weight).max()
    bias_diff = np.abs(layer.bias.grad - num_grad_bias).max()

    # set a comparison threshold
    THRESHOLD = 5e-3

    assert weight_diff < THRESHOLD, f"Weight gradient mismatch: {weight_diff}"
    assert bias_diff < THRESHOLD, f"Bias gradient mismatch: {bias_diff}"

    # Check input gradient shape
    assert (
        grad_input.shape == x.shape
    ), f"Input gradient shape mismatch: {grad_input.shape} != {x.shape}"
