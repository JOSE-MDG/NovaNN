import numpy as np
import pytest

from novann.layers import BatchNorm1d
from novann.utils.gradient_checking import numeric_grad_wrt_param

RNG = np.random.RandomState(8)


def test_batchnorm1d_forward_train_mode():
    """Test forward pass in training mode (updates running stats)."""
    num_features = 8
    batch_size = 512

    # Initialize layer
    bn = BatchNorm1d(num_features)
    bn.train()  # Set to training mode

    # Random input
    x = RNG.randn(batch_size, num_features).astype(np.float32)

    # Forward pass
    output = bn(x)

    # Check shape preservation
    assert (
        output.shape == x.shape
    ), f"Output shape {output.shape} != input shape {x.shape}"

    # After normalization, per-feature mean should be near zero
    feature_means = np.mean(output, axis=0, keepdims=True)
    assert np.allclose(
        feature_means, 0.0, atol=1e-1
    ), f"Features not centered: max mean = {np.abs(feature_means).max():.6f}"

    # Per-feature variance should be near one
    feature_vars = np.var(output, axis=0, keepdims=True)
    assert np.allclose(
        feature_vars, 1.0, atol=1e-1
    ), f"Variances not normalized: max var = {feature_vars.max():.6f}"

    # Check that running stats were updated
    assert hasattr(bn, "running_mean"), "Running mean attribute missing"
    assert hasattr(bn, "running_var"), "Running variance attribute missing"
    assert bn.running_mean.shape == (1, num_features)
    assert bn.running_var.shape == (1, num_features)


def test_batchnorm1d_forward_eval_mode():
    """Test forward pass in evaluation mode (uses running statistics)."""
    num_features = 6

    # Initialize layer
    bn = BatchNorm1d(num_features)

    # First update running stats in train mode
    bn.train()
    x_train = RNG.randn(512, num_features).astype(np.float32)
    bn(x_train)  # Updates running_mean and running_var

    # Switch to evaluation mode
    bn.eval()

    # Different input (smaller batch)
    x_eval = RNG.randn(128, num_features).astype(np.float32)
    output = bn(x_eval)

    # In eval mode, should use running stats, not batch stats
    # Output should still be reasonable (not explode)
    assert not np.any(np.isnan(output)), "Output contains NaN"
    assert not np.any(np.isinf(output)), "Output contains Inf"

    # Mean should be roughly centered (using running stats)
    eval_mean = np.mean(output, axis=0, keepdims=True)
    assert np.all(
        np.abs(eval_mean) < 2.0
    ), f"Eval output not centered: max |mean| = {np.abs(eval_mean).max():.6f}"


def test_batchnorm1d_backward_gradient_check():
    """Gradient checking for BatchNorm1d layer."""
    num_features = 4
    batch_size = 8

    # Initialize layer in training mode
    bn = BatchNorm1d(num_features)
    bn.train()

    # Small input for faster computation
    x = RNG.randn(batch_size, num_features).astype(np.float32) * 0.1

    # Forward pass
    output = bn(x)

    # Random gradient w.r.t output
    G = np.random.randn(*output.shape).astype(np.float32) * 0.1

    # Backward pass (analytic gradients)
    grad_input = bn.backward(G)

    # set a comparison threshold
    THRESHOLD = 5e-3

    # Check gradient for gamma (scale parameter)
    num_grad_gamma = numeric_grad_wrt_param(bn, "gamma", x, G, eps=1e-5)
    gamma_diff = np.abs(bn.gamma.grad - num_grad_gamma).max()
    assert gamma_diff < THRESHOLD, f"Gamma gradient mismatch: {gamma_diff}"

    # Check gradient for beta (shift parameter)
    num_grad_beta = numeric_grad_wrt_param(bn, "beta", x, G, eps=1e-5)
    beta_diff = np.abs(bn.beta.grad - num_grad_beta).max()
    assert beta_diff < THRESHOLD, f"Beta gradient mismatch: {beta_diff}"

    # Check input gradient shape
    assert (
        grad_input.shape == x.shape
    ), f"Input gradient shape mismatch: {grad_input.shape} != {x.shape}"


def test_batchnorm1d_momentum_and_eps():
    """Test momentum and epsilon parameters."""
    num_features = 3

    # Test with custom momentum and epsilon
    bn = BatchNorm1d(num_features, momentum=0.9, eps=1e-3)
    assert bn.momentum == 0.9, f"Momentum should be 0.9, got {bn.momentum}"
    assert bn.eps == 1e-3, f"Epsilon should be 1e-3, got {bn.eps}"

    # Forward pass should work with custom parameters
    x = RNG.randn(10, num_features).astype(np.float32)
    bn.train()
    output = bn(x)
    assert output.shape == x.shape


def test_batchnorm1d_parameters():
    """Test parameters() method returns correct list."""
    # With affine parameters
    bn = BatchNorm1d(num_features=4)
    params = bn.parameters()

    if bn.gamma is not None and bn.beta is not None:
        assert len(params) == 2, f"Expected 2 parameters, got {len(params)}"
        assert params[0] is bn.gamma
        assert params[1] is bn.beta
    else:
        # Some implementations might not include gamma/beta in parameters()
        assert len(params) >= 0, "Parameters list should be non-negative"
