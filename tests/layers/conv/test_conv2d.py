import numpy as np
import pytest
import novann as nn

from novann.utils.gradient_checking import numeric_grad_wrt_param

RNG = np.random.RandomState(0)


def test_conv2d_forward_shape():
    """Test forward pass output shape."""
    layer = nn.Conv2d(
        in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True
    )

    # Input shape: (batch_size, channels, height, width)
    x = RNG.randn(4, 3, 32, 32).astype(np.float32)
    output = layer(x)

    # Expected output: H_out = floor((H + 2*padding - KH) / stride) + 1
    # = floor((32 + 2*1 - 3) / 1) + 1 = 32
    assert output.shape == (
        4,
        16,
        32,
        32,
    ), f"Expected (4, 16, 32, 32), got {output.shape}"
    assert output.dtype == np.float32


def test_conv2d_forward_no_bias():
    """Test forward pass without bias."""
    layer = nn.Conv2d(
        in_channels=2, out_channels=4, kernel_size=5, stride=2, padding=0, bias=False
    )

    x = RNG.randn(2, 2, 20, 20).astype(np.float32)
    output = layer(x)

    assert layer.bias is None, "Bias should be None when bias=False"

    # H_out = floor((20 + 0 - 5) / 2) + 1 = 8
    assert output.shape == (2, 4, 8, 8), f"Expected (2, 4, 8, 8), got {output.shape}"


def test_conv2d_backward_gradient_check_small():
    """Gradient checking for Conv2d layer with small input."""

    layer = nn.Conv2d(
        in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True
    )

    # Very small input for faster computation
    x = RNG.randn(2, 2, 6, 6).astype(np.float32) * 0.1

    # Forward pass
    output = layer(x)

    # Random gradient w.r.t output
    G = RNG.randn(*output.shape).astype(np.float32) * 0.1

    # Backward pass (analytic gradients)
    grad_input = layer.backward(G)

    # Numerical gradient for weight (expensive, so use small eps)
    num_grad_weight = numeric_grad_wrt_param(layer, "weight", x, G, eps=1e-5)

    # Set a comparison threshold
    THRESHOLD = 5e-3

    # Compare analytic vs numerical gradients with relaxed tolerance
    weight_diff = np.abs(layer.weight.grad - num_grad_weight).max()
    assert weight_diff < THRESHOLD, f"Weight gradient mismatch: {weight_diff}"

    # Check bias gradient if present
    if layer.bias is not None:
        num_grad_bias = numeric_grad_wrt_param(layer, "bias", x, G, eps=1e-5)
        bias_diff = np.abs(layer.bias.grad - num_grad_bias).max()
        assert bias_diff < THRESHOLD, f"Bias gradient mismatch: {bias_diff}"

    # Check input gradient shape
    assert (
        grad_input.shape == x.shape
    ), f"Input gradient shape mismatch: {grad_input.shape} != {x.shape}"


def test_conv2d_different_kernel_stride_padding():
    """Test Conv2d with various kernel sizes, strides, and paddings."""
    test_cases = [
        {"kernel": 3, "stride": 1, "padding": 0},
        {"kernel": 5, "stride": 2, "padding": 1},
        {"kernel": (3, 5), "stride": (2, 1), "padding": (1, 2)},
    ]

    for config in test_cases:
        layer = nn.Conv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=config["kernel"],
            stride=config["stride"],
            padding=config["padding"],
            bias=False,
        )

        x = RNG.randn(3, 2, 16, 16).astype(np.float32)
        output = layer(x)

        # Basic shape checks
        assert output.shape[0] == 3  # batch size preserved
        assert output.shape[1] == 4  # output channels correct
        assert output.shape[2] > 0 and output.shape[3] > 0  # positive dimensions


def test_conv2d_padding_modes():
    """Test different padding modes."""
    # Test basic padding modes
    for padding_mode in ("zeros", "reflect", "replicate", "circular"):
        layer = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            padding=2,
            padding_mode=padding_mode,
            bias=False,
        )

        x = np.ones((1, 1, 8, 8), dtype=np.float32)
        output = layer(x)

        # Just check it runs without error
        assert output.shape[0] == 1  # batch size preserved
        assert output.shape[1] == 2  # output channels correct


def test_conv2d_parameters():
    """Test parameters() method returns correct list."""
    # With bias
    layer_with_bias = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, bias=True)
    params = layer_with_bias.parameters()
    assert len(params) == 2, f"Expected 2 parameters, got {len(params)}"
    assert params[0] is layer_with_bias.weight
    assert params[1] is layer_with_bias.bias

    # Without bias
    layer_no_bias = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, bias=False)
    params = layer_no_bias.parameters()
    assert len(params) == 1, f"Expected 1 parameter, got {len(params)}"
    assert params[0] is layer_no_bias.weight
