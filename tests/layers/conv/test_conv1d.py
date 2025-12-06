import numpy as np
from novann.layers.convolutional.conv import Conv1d
from novann.utils.gradient_checking.numerical import (
    numeric_grad_wrt_param,
    numeric_grad_scalar_wrt_x,
)

RNG = np.random.RandomState(8)


def test_conv1d_forward_shape():
    """Test forward pass output shape."""
    layer = Conv1d(
        in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True
    )

    # Input shape: (batch_size, channels, length)
    x = RNG.randn(4, 3, 32).astype(np.float32)
    output = layer(x)

    # Expected output length: L_out = floor((L + 2*padding - K) / stride) + 1
    # = floor((32 + 2*1 - 3) / 1) + 1 = 32
    assert output.shape == (4, 16, 32), f"Expected (4, 16, 32), got {output.shape}"
    assert output.dtype == np.float32


def test_conv1d_forward_no_bias():
    """Test forward pass without bias."""
    layer = Conv1d(
        in_channels=2, out_channels=4, kernel_size=5, stride=2, padding=0, bias=False
    )

    x = RNG.randn(2, 2, 20).astype(np.float32)
    output = layer(x)

    assert layer.bias is None, "Bias should be None when bias=False"

    # L_out = floor((20 + 0 - 5) / 2) + 1 = 8
    assert output.shape == (2, 4, 8), f"Expected (2, 4, 8), got {output.shape}"


def test_conv1d_backward_gradient_check():
    """Gradient checking for Conv1d layer parameters."""

    layer = Conv1d(
        in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True
    )

    # Small input for faster computation
    x = np.random.randn(2, 2, 8).astype(np.float32) * 0.1

    # Forward pass
    output = layer(x)

    # Random gradient w.r.t output
    G = np.random.randn(*output.shape).astype(np.float32) * 0.1

    # Backward pass (analytic gradients)
    grad_input = layer.backward(G)

    # Numerical gradient for weight
    num_grad_weight = numeric_grad_wrt_param(layer, "weight", x, G, eps=1e-5)

    # Set a comparison threshold
    THRESHOLD = 1e-3

    # Compare analytic vs numerical gradients
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


def test_conv1d_padding_modes():
    """Test different padding modes."""
    for padding_mode in ("zeros", "reflect", "replicate", "circular"):
        layer = Conv1d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            padding=2,
            padding_mode=padding_mode,
            bias=False,
        )

        x = np.ones((1, 1, 5), dtype=np.float32)
        output = layer(x)

        # Just check it runs without error
        assert output.shape[0] == 1  # batch size preserved
        assert output.shape[1] == 2  # output channels correct


def test_conv1d_parameters():
    """Test parameters() method returns correct list."""
    # With bias
    layer_with_bias = Conv1d(in_channels=3, out_channels=5, kernel_size=3, bias=True)
    params = layer_with_bias.parameters()
    assert len(params) == 2, f"Expected 2 parameters, got {len(params)}"
    assert params[0] is layer_with_bias.weight
    assert params[1] is layer_with_bias.bias

    # Without bias
    layer_no_bias = Conv1d(in_channels=3, out_channels=5, kernel_size=3, bias=False)
    params = layer_no_bias.parameters()
    assert len(params) == 1, f"Expected 1 parameter, got {len(params)}"
    assert params[0] is layer_no_bias.weight
