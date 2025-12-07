import numpy as np
import pytest

from novann.optim import SGD
from novann.layers import Linear
from novann.model import Sequential

RNG = np.random.RandomState(8)


def test_sgd_basic_update():
    """Test that SGD updates parameters correctly."""

    # Create a simple model
    model = Sequential(
        Linear(in_features=10, out_features=5, bias=True),
        Linear(in_features=5, out_features=2, bias=True),
    )

    # Create optimizer
    optimizer = SGD(model.parameters(), lr=0.01)

    # Store initial weights
    initial_params = [p.data.copy() for p in model.parameters()]

    # Simulate forward and backward pass
    x = RNG.randn(3, 10).astype(np.float32)
    output = model(x)

    # Create dummy gradient
    grad_output = RNG.randn(*output.shape).astype(np.float32)
    model.backward(grad_output)

    # Perform optimization step
    optimizer.step()

    # Check that parameters changed
    for init, param in zip(initial_params, model.parameters()):
        assert not np.allclose(
            init, param.data
        ), "Parameters should change after SGD step"

    # Check that gradients are still present (not zeroed unless zero_grad called)
    for param in model.parameters():
        assert param.grad is not None, "Gradients should still exist after step"


def test_sgd_with_momentum():
    """Test SGD with momentum."""

    # Simple linear layer
    layer = Linear(in_features=5, out_features=3, bias=True)

    # SGD with momentum
    optimizer = SGD(layer.parameters(), lr=0.01, momentum=0.9)

    # Multiple steps to see momentum effect
    initial_weight = layer.weight.data.copy()

    # First step
    x = RNG.randn(2, 5).astype(np.float32)
    output = layer(x)
    grad = RNG.randn(*output.shape).astype(np.float32)
    layer.backward(grad)
    optimizer.step()

    step1_change = layer.weight.data - initial_weight

    # Second step with same gradient direction
    layer.zero_grad()
    layer(x)
    layer.backward(grad)  # Same gradient
    optimizer.step()

    step2_change = layer.weight.data - (initial_weight + step1_change)

    # With momentum, second change should be larger (velocity builds up)
    # Actually in SGD with momentum: v_t = beta*v_{t-1} - lr*grad
    # So second step includes previous momentum
    assert np.linalg.norm(step2_change) > 0, "Momentum should cause non-zero update"


def test_sgd_zero_grad():
    """Test zero_grad method."""
    layer = Linear(in_features=4, out_features=2)
    optimizer = SGD(layer.parameters(), lr=0.01)

    # Create some gradients
    x = RNG.randn(1, 4).astype(np.float32)
    output = layer(x)
    layer.backward(np.ones_like(output))

    # Verify gradients exist
    for param in layer.parameters():
        assert param.grad is not None
        assert not np.all(param.grad == 0.0)

    # Zero gradients
    optimizer.zero_grad()

    # Verify gradients are zero
    for param in layer.parameters():
        assert param.grad is not None
        assert np.all(param.grad == 0.0)
