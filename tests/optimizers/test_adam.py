import numpy as np
import pytest

from novann.optim import Adam
from novann.layers import Linear, Conv2d

RNG = np.random.RandomState(8)


def test_adam_basic_update():
    """Test that Adam updates parameters."""

    # Test with Linear layer
    linear = Linear(in_features=8, out_features=4, bias=True)
    optimizer = Adam(linear.parameters(), lr=0.01)

    initial_weight = linear.weight.data.copy()

    # Forward and backward
    x = RNG.randn(2, 8).astype(np.float32)
    output = linear(x)
    grad = RNG.randn(*output.shape).astype(np.float32)
    linear.backward(grad)

    # Optimizer step
    optimizer.step()

    # Check update
    assert not np.allclose(
        initial_weight, linear.weight.data
    ), "Parameters should change after Adam step"

    # Check optimizer state
    assert optimizer.t == 1, "Step counter should increment"


def test_adam_with_conv_layer():
    """Test Adam with convolutional layers."""

    conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=True)
    optimizer = Adam(conv.parameters(), lr=0.001, betas=(0.9, 0.999))

    initial_weight = conv.weight.data.copy()

    # Forward and backward for conv layer
    x = RNG.randn(2, 3, 16, 16).astype(np.float32)
    output = conv(x)
    grad = RNG.randn(*output.shape).astype(np.float32)
    conv.backward(grad)

    # Optimizer step
    optimizer.step()

    # Check update happened
    assert not np.allclose(
        initial_weight, conv.weight.data
    ), "Conv layer parameters should change after Adam step"

    # Check bias also updated
    assert conv.bias is not None
    assert not np.all(conv.bias.grad == 0.0)


def test_adam_bias_correction():
    """Test Adam's bias correction mechanism."""

    layer = Linear(in_features=5, out_features=2)
    optimizer = Adam(layer.parameters(), lr=0.01, betas=(0.9, 0.999))

    # Take multiple steps to see bias correction effect
    changes = []

    for step in range(3):
        x = RNG.randn(1, 5).astype(np.float32)
        output = layer(x)
        grad = RNG.randn(*output.shape).astype(np.float32)

        layer.zero_grad()
        layer.backward(grad)

        weight_before = layer.weight.data.copy()
        optimizer.step()
        weight_after = layer.weight.data.copy()

        changes.append(np.linalg.norm(weight_after - weight_before))

    # With bias correction, early steps might have different magnitudes
    # but all should be non-zero
    assert all(change > 0 for change in changes), "All steps should cause updates"
    assert optimizer.t == 3, "Should have taken 3 steps"
