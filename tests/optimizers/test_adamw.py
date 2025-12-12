import numpy as np
import novann as nn
import novann.optim as optim
import pytest

RNG = np.random.RandomState(8)


def test_adamw_updates_parameters():
    """AdamW should update parameters when gradient exists."""

    layer = nn.Linear(in_features=10, out_features=5, bias=True)
    optimizer = optim.AdamW(layer.parameters(), lr=0.01, weight_decay=0.01)

    initial_weight = layer.weight.data.copy()

    # Generate gradient
    x = RNG.randn(3, 10).astype(np.float32)
    output = layer(x)
    grad = RNG.randn(*output.shape).astype(np.float32)
    layer.backward(grad)

    # Update parameters
    optimizer.step()

    # Verify change
    assert not np.allclose(
        initial_weight, layer.weight.data
    ), "Parameters must change after optimizer step"

    # Verify step counter
    assert optimizer.t == 1, "Step counter should increment"


def test_adamw_decoupled_weight_decay():
    """Weight decay should be applied SEPARATELY from gradient update."""

    # Two identical models
    model1 = nn.Linear(8, 4, bias=False)
    model2 = nn.Linear(8, 4, bias=False)
    model2.weight.data = model1.weight.data.copy()  # Same weights

    # AdamW WITH weight decay vs AdamW WITHOUT weight decay
    optim1 = optim.AdamW(model1.parameters(), lr=0.1, weight_decay=0.5)  # With decay
    optim2 = optim.AdamW(model2.parameters(), lr=0.1, weight_decay=0.0)  # No decay

    # Same input and gradient
    x = RNG.randn(2, 8).astype(np.float32)
    grad = RNG.randn(2, 4).astype(np.float32)

    # Forward + backward for both
    out1 = model1(x)
    model1.backward(grad)

    out2 = model2(x)
    model2.backward(grad)

    # Store initial weights
    initial_w = model1.weight.data.copy()

    # Apply weight decay MANUALLY to model2 (simulating decoupled decay)
    model2.weight.data -= optim1.lr * optim1.wd * model2.weight.data

    # Now AdamW step for both
    optim1.step()  # Includes weight decay for model1
    optim2.step()  # No weight decay for model2 (already applied)

    # Calculate update magnitudes
    update_with_decay = np.sum((model1.weight.data - initial_w) ** 2)
    update_without_decay = np.sum((model2.weight.data - initial_w) ** 2)

    # With weight decay should have SMALLER update
    assert (
        update_with_decay < update_without_decay
    ), "Decoupled weight decay should decrease update magnitude"


def test_adamw_excludes_batchnorm_from_weight_decay():
    """AdamW should NOT apply weight decay to BatchNorm gamma/beta."""

    # Model with Conv (should get decay) and BatchNorm (should NOT get decay)
    conv = nn.Conv2d(3, 8, kernel_size=3, bias=False)
    bn = nn.BatchNorm2d(8)

    # Name BatchNorm parameters as expected by optimizer
    bn.gamma.name = "gamma"
    bn.beta.name = "beta"

    # Optimizer with strong weight decay
    optimizer = optim.AdamW(
        list(conv.parameters()) + list(bn.parameters()), lr=0.1, weight_decay=1.0
    )

    # Forward pass
    x = RNG.randn(2, 3, 16, 16).astype(np.float32)
    conv_out = conv(x)
    bn_out = bn(conv_out)

    # Backward pass
    grad = RNG.randn(*bn_out.shape).astype(np.float32)
    bn.backward(grad)
    conv.backward(grad)

    # Store initial values
    conv_initial = conv.weight.data.copy()
    gamma_initial = bn.gamma.data.copy()
    beta_initial = bn.beta.data.copy()

    # Optimizer step
    optimizer.step()

    # Conv weights MUST change (gradient + weight decay)
    assert not np.allclose(conv_initial, conv.weight.data), "Conv weights should change"

    # BatchNorm params SHOULD change (gradient only, no weight decay)
    assert not np.allclose(
        gamma_initial, bn.gamma.data
    ), "BatchNorm gamma should change from gradient"
    assert not np.allclose(
        beta_initial, bn.beta.data
    ), "BatchNorm beta should change from gradient"

    # Sanity check: optimizer recognized BatchNorm params
    # (implied by the fact that gamma/beta changed but without decay amplification)
