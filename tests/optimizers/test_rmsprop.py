import numpy as np
import novann as nn
import novann.optim as optim
import pytest


RNG = np.random.RandomState(9)


def test_rmsprop_basic_update():
    """Test that RMSprop updates parameters."""

    layer = nn.Linear(in_features=6, out_features=3, bias=True)
    optimizer = optim.RMSprop(layer.parameters(), lr=0.01, beta=0.9)

    initial_weight = layer.weight.data.copy()

    # Forward and backward
    x = RNG.randn(2, 6).astype(np.float32)
    output = layer(x)
    grad = RNG.randn(*output.shape).astype(np.float32)
    layer.backward(grad)

    # Optimizer step
    optimizer.step()

    # Check parameter update
    assert not np.allclose(
        initial_weight, layer.weight.data
    ), "Parameters should change after RMSprop step"

    # Check that optimizer has moments
    assert hasattr(optimizer, "s_t"), "RMSprop should have moments 's_t' attribute"
    assert len(optimizer.s_t) == len(layer.parameters())


def test_rmsprop_with_weight_decay():
    """Test RMSprop with weight decay."""

    # Two identical layers, one with weight decay, one without
    layer1 = nn.Linear(in_features=5, out_features=2)
    layer2 = nn.Linear(in_features=5, out_features=2)

    # Copy same initial weights
    layer2.weight.data = layer1.weight.data.copy()
    layer2.bias.data = layer1.bias.data.copy()

    # Optimizers
    opt1 = optim.RMSprop(layer1.parameters(), lr=0.01, weight_decay=0.1)  # L2
    opt2 = optim.RMSprop(layer2.parameters(), lr=0.01, weight_decay=0.0)  # no decay

    # Same forward pass
    x = RNG.randn(1, 5).astype(np.float32)

    output1 = layer1(x)
    output2 = layer2(x)

    # Same gradient
    grad = np.ones_like(output1)
    layer1.backward(grad.copy())
    layer2.backward(grad.copy())

    # Optimization steps
    opt1.step()
    opt2.step()

    # With weight decay, parameters should be smaller
    # (L2 decay adds wd * param to gradient, pushing toward zero)
    norm1 = np.linalg.norm(layer1.weight.data)
    norm2 = np.linalg.norm(layer2.weight.data)

    assert (
        norm1 < norm2
    ), "L2 Weight decay (RMSprop) should result in a smaller weight magnitude"


def test_rmsprop_zero_grad():
    """Test zero_grad method for RMSprop."""

    layer = nn.Linear(in_features=3, out_features=1)
    optimizer = optim.RMSprop(layer.parameters(), lr=0.01)

    # Create gradients
    x = RNG.randn(2, 3).astype(np.float32)
    output = layer(x)
    layer.backward(np.ones_like(output))

    # Zero gradients
    optimizer.zero_grad()

    # Check all gradients are zero
    for param in layer.parameters():
        assert param.grad is not None
        assert np.all(param.grad == 0.0)
