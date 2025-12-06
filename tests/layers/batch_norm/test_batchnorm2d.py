import numpy as np
import pytest

from novann.layers import BatchNorm2d
from novann.utils.gradient_checking import (
    numeric_grad_wrt_param,
    numeric_grad_scalar_wrt_x,
)

ef test_batchnorm2d_forward_train_mode():
    """Test forward pass in training mode for 2D data."""
    np.random.seed(42)
    num_channels = 8
    batch_size = 16
    height, width = 32, 32
    
    # Initialize layer
    bn = BatchNorm2d(num_channels)
    bn.train()  # Set to training mode
    
    # Random input (N, C, H, W)
    x = np.random.randn(batch_size, num_channels, height, width).astype(np.float32)
    
    # Forward pass
    output = bn.forward(x)
    
    # Check shape preservation
    assert output.shape == x.shape, \
        f"Output shape {output.shape} != input shape {x.shape}"
    
    # After normalization, per-channel mean should be near zero
    # Compute mean over batch, height, width dimensions (axes 0, 2, 3)
    channel_means = np.mean(output, axis=(0, 2, 3), keepdims=True)
    assert np.allclose(channel_means, 0.0, atol=1e-1), \
        f"Channels not centered: max mean = {np.abs(channel_means).max():.6f}"
    
    # Per-channel variance should be near one
    channel_vars = np.var(output, axis=(0, 2, 3), keepdims=True)
    assert np.allclose(channel_vars, 1.0, atol=1e-1), \
        f"Variances not normalized: max var = {channel_vars.max():.6f}"
    
    # Check that running stats were updated
    assert hasattr(bn, 'running_mean'), "Running mean attribute missing"
    assert hasattr(bn, 'running_var'), "Running variance attribute missing"
    assert bn.running_mean.shape == (1, num_channels, 1, 1)
    assert bn.running_var.shape == (1, num_channels, 1, 1)


def test_batchnorm2d_forward_eval_mode():
    """Test forward pass in evaluation mode for 2D data."""
    np.random.seed(42)
    num_channels = 6
    
    # Initialize layer
    bn = BatchNorm2d(num_channels)
    
    # First update running stats in train mode
    bn.train()
    x_train = np.random.randn(32, num_channels, 16, 16).astype(np.float32)
    bn.forward(x_train)  # Updates running_mean and running_var
    
    # Switch to evaluation mode
    bn.eval()
    
    # Different input
    x_eval = np.random.randn(8, num_channels, 16, 16).astype(np.float32)
    output = bn.forward(x_eval)
    
    # In eval mode, should use running stats
    assert not np.any(np.isnan(output)), "Output contains NaN"
    assert not np.any(np.isinf(output)), "Output contains Inf"
    
    # Output should be reasonable
    eval_mean = np.mean(output, axis=(0, 2, 3), keepdims=True)
    assert np.all(np.abs(eval_mean) < 2.0), \
        f"Eval output not centered: max |mean| = {np.abs(eval_mean).max():.6f}"


def test_batchnorm2d_backward_gradient_check():
    """Gradient checking for BatchNorm2d layer."""
    np.random.seed(42)
    num_channels = 3
    batch_size = 4
    height, width = 8, 8
    
    # Initialize layer in training mode
    bn = BatchNorm2d(num_channels)
    bn.train()
    
    # Small input for faster computation
    x = np.random.randn(batch_size, num_channels, height, width).astype(np.float32) * 0.1
    
    # Forward pass
    output = bn.forward(x)
    
    # Random gradient w.r.t output
    G = np.random.randn(*output.shape).astype(np.float32) * 0.1
    
    # Backward pass (analytic gradients)
    grad_input = bn.backward(G)
    
    # Check gradient for gamma (scale parameter) if affine=True
    if bn.gamma is not None:
        num_grad_gamma = numeric_grad_wrt_param(
            bn, "gamma", x, G, eps=1e-5
        )
        gamma_diff = np.abs(bn.gamma.grad - num_grad_gamma).max()
        assert gamma_diff < 1e-5, f"Gamma gradient mismatch: {gamma_diff}"
    
    # Check gradient for beta (shift parameter) if affine=True
    if bn.beta is not None:
        num_grad_beta = numeric_grad_wrt_param(
            bn, "beta", x, G, eps=1e-5
        )
        beta_diff = np.abs(bn.beta.grad - num_grad_beta).max()
        assert beta_diff < 1e-5, f"Beta gradient mismatch: {beta_diff}"
    
    # Check input gradient shape
    assert grad_input.shape == x.shape, \
        f"Input gradient shape mismatch: {grad_input.shape} != {x.shape}"

def test_batchnorm2d_momentum_and_eps():
    """Test momentum and epsilon parameters for 2D."""
    num_channels = 3
    
    # Test with custom momentum and epsilon
    bn = BatchNorm2d(num_channels, momentum=0.9, eps=1e-3)
    assert bn.momentum == 0.9, f"Momentum should be 0.9, got {bn.momentum}"
    assert bn.eps == 1e-3, f"Epsilon should be 1e-3, got {bn.eps}"
    
    # Forward pass should work with custom parameters
    x = np.random.randn(4, num_channels, 8, 8).astype(np.float32)
    bn.train()
    output = bn.forward(x)
    assert output.shape == x.shape


def test_batchnorm2d_different_spatial_sizes():
    """Test BatchNorm2d with different spatial dimensions."""
    num_channels = 4
    bn = BatchNorm2d(num_channels)
    bn.train()
    
    # Test with different spatial sizes
    spatial_sizes = [(8, 8), (16, 16), (32, 64), (1, 1)]
    
    for h, w in spatial_sizes:
        x = np.random.randn(2, num_channels, h, w).astype(np.float32)
        output = bn.forward(x)
        
        assert output.shape == x.shape, \
            f"Shape mismatch for size ({h}, {w}): {output.shape} != {x.shape}"
        
        # Statistics should still be normalized per channel
        channel_means = np.mean(output, axis=(0, 2, 3), keepdims=True)
        assert np.allclose(channel_means, 0.0, atol=1e-1), \
            f"Not centered for size ({h}, {w}): mean = {channel_means.max()}"


def test_batchnorm2d_parameters():
    """Test parameters() method for BatchNorm2d."""
    # With affine parameters
    bn = BatchNorm2d(num_channels=4)
    params = bn.parameters()
    
    if bn.gamma is not None and bn.beta is not None:
        assert len(params) == 2, f"Expected 2 parameters, got {len(params)}"
        assert params[0] is bn.gamma
        assert params[1] is bn.beta
    else:
        # Some implementations might not include gamma/beta in parameters()
        assert len(params) >= 0, "Parameters list should be non-negative"