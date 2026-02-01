import pytest
import nova
import nova.nn.functional as F
from nova.utils import grad_check_wrt_inputs
from nova.nn import init
from nova.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
)

nova.manual_seed(8)


class TestBatchNormForward:
    """Test forward pass and basic normalization."""

    def test_batchnorm1d_2d_input(self):
        """Test BatchNorm1d with 2D input (N, C)."""
        bn = BatchNorm1d(num_features=10)
        bn.train()
        x = nova.randn(4, 10)
        y = bn(x)

        assert y.shape == x.shape
        # Check that output is normalized (approximately zero mean, unit variance)
        assert abs(y.mean().item()) < 0.1
        assert abs(y.std().item() - 1.0) < 0.2

    def test_batchnorm1d_3d_input(self):
        """Test BatchNorm1d with 3D input (N, C, L)."""
        bn = BatchNorm1d(num_features=16)
        bn.train()
        x = nova.randn(8, 16, 20)
        y = bn(x)

        assert y.shape == x.shape

    def test_batchnorm2d_forward(self):
        """Test BatchNorm2d with 4D input (N, C, H, W)."""
        bn = BatchNorm2d(num_features=32)
        bn.train()
        x = nova.randn(4, 32, 16, 16)
        y = bn(x)

        assert y.shape == x.shape
        # Verify normalization per channel
        for c in range(32):
            channel_output = y[:, c, :, :]
            assert abs(channel_output.mean().item()) < 0.1

    def test_batchnorm3d_forward(self):
        """Test BatchNorm3d with 5D input (N, C, D, H, W)."""
        bn = BatchNorm3d(num_features=16)
        bn.train()
        x = nova.randn(2, 16, 8, 8, 8)
        y = bn(x)

        assert y.shape == x.shape


class TestBatchNormBackward:
    """Test batchnorm backward and gradients"""

    def test_batchnorm_grad(self):
        """test batchnorm gradients"""
        bn = BatchNorm1d(10)
        x = nova.randn(32, 10, requires_grad=True)
        y = nova.randint(0, 10, (32,))  # make a dummy labels for the loss

        assert x.grad is None

        # Forward
        out = bn(x)
        loss = F.cross_entropy(out, y)

        assert loss.numel() == 1, f"{loss.numel()} > 1"

        # Compute gradients wrt input (x)
        gradients = nova.autograd.grad(loss, x, create_graph=True)

        assert x.grad is not None
        assert nova.allclose(gradients[0], x.grad, rtol=5e-2, atol=5e-2)

    def test_batchnorm1d_backward(self):
        """Test gradients with finite differences"""
        bn = BatchNorm1d(4)
        x = nova.randn(8, 4, requires_grad=True)

        analitycal, numerical = grad_check_wrt_inputs(bn, x, eps=1e-4)

        assert nova.allclose(analitycal[0], numerical[0], rtol=1e-2, atol=5e-2)

    def test_batchnorm2d_backward(self):
        """Test gradients with finite differences"""
        bn = BatchNorm2d(3)
        x = nova.randn(8, 3, 3, 3, requires_grad=True)

        analitycal, numerical = grad_check_wrt_inputs(bn, x, eps=1e-4)

        assert nova.allclose(analitycal[0], numerical[0], rtol=1e-2, atol=5e-2)

    def test_batchnorm3d_backward(self):
        """Test gradients with finite differences"""
        bn = BatchNorm3d(3)
        x = nova.randn(8, 3, 3, 3, 3, requires_grad=True)

        analitycal, numerical = grad_check_wrt_inputs(bn, x, eps=1e-4)

        assert nova.allclose(analitycal[0], numerical[0], rtol=1e-2, atol=5e-2)


class TestBatchNormRunningStats:
    """Test running statistics tracking."""

    def test_running_stats_update_during_training(self):
        """Verify running_mean and running_var update in training mode."""
        bn = BatchNorm1d(num_features=10, momentum=0.1)
        bn.train()

        # Initial values
        assert nova.allclose(bn.running_mean, nova.zeros((10,)))
        assert nova.allclose(bn.running_var, nova.ones((10,)))

        # Forward pass
        x = nova.randn(32, 10) * 2.0 + 5.0  # mean~5, std~2
        _ = bn(x)

        # Running stats should have moved towards batch statistics
        assert not nova.allclose(bn.running_mean, nova.zeros((10,)))
        assert not nova.allclose(bn.running_var, nova.ones((10,)))
        # Should be close to batch statistics (dampened by momentum)
        assert bn.running_mean.mean().item() > 0.1  # Moved from 0 towards 5

    def test_running_stats_not_updated_in_eval(self):
        """Verify running stats don't update in eval mode."""
        bn = BatchNorm1d(num_features=10)
        bn.eval()

        running_mean_before = bn.running_mean.clone()
        running_var_before = bn.running_var.clone()

        x = nova.randn(32, 10) * 10.0 + 20.0  # Very different statistics
        _ = bn(x)

        # Running stats should be unchanged
        assert nova.allclose(bn.running_mean, running_mean_before)
        assert nova.allclose(bn.running_var, running_var_before)

    def test_eval_mode_uses_running_stats(self):
        """Verify eval mode uses running stats instead of batch stats."""
        bn = BatchNorm2d(num_features=8, momentum=0.5)
        bn.train()

        # Train on some data to populate running stats
        x_train = nova.randn(16, 8, 4, 4) * 2.0 + 3.0
        _ = bn(x_train)

        # Switch to eval and test with very different data
        bn.eval()
        x_eval = nova.randn(16, 8, 4, 4) * 10.0 - 50.0  # Very different stats
        y_eval = bn(x_eval)

        # Output should be normalized using running stats, not batch stats
        # So it won't have zero mean (because input stats differ from running stats)
        assert y_eval.shape == x_eval.shape


class TestBatchNormAffineParameters:
    """Test affine transformation (weight and bias)."""

    def test_affine_parameters_applied(self):
        """Verify that weight and bias are correctly applied."""
        bn = BatchNorm1d(num_features=5, affine=True)
        bn.train()

        # Set specific weight and bias
        init.constant_(bn.weight, 2.0)
        init.constant_(bn.bias, 0.5)

        x = nova.randn(32, 5)
        y = bn(x)

        # After normalization (mean~0, std~1), apply weight*x + bias
        # So output should have values around bias ± weight
        assert abs(y.mean().item() - 0.5) < 0.3  # Close to bias

    def test_no_affine_parameters(self):
        """Verify behavior when affine=False."""
        bn = BatchNorm1d(num_features=10, affine=False)

        assert bn.weight is None
        assert bn.bias is None

        bn.train()
        x = nova.randn(32, 10)
        y = bn(x)

        # Should still normalize without affine transform
        assert y.shape == x.shape


class TestBatchNormDimensions:
    """Test dimension validation."""

    def test_batchnorm1d_rejects_wrong_dims(self):
        """BatchNorm1d should only accept 2D or 3D input."""
        bn = BatchNorm1d(num_features=10)

        # Should work with 2D and 3D
        bn(nova.randn(4, 10))
        bn(nova.randn(4, 10, 20))

        # Should fail with other dimensions
        with pytest.raises(ValueError, match="expected 2D or 3D input"):
            bn(nova.randn(4, 10, 20, 30))  # 4D

        with pytest.raises(ValueError, match="expected 2D or 3D input"):
            bn(nova.randn(10))  # 1D

    def test_batchnorm2d_requires_4d_input(self):
        """BatchNorm2d should only accept 4D input."""
        bn = BatchNorm2d(num_features=16)

        # Should work with 4D
        bn(nova.randn(4, 16, 8, 8))

        # Should fail with other dimensions
        with pytest.raises(ValueError, match="expected 4D input"):
            bn(nova.randn(4, 16, 8))  # 3D

    def test_batchnorm3d_requires_5d_input(self):
        """BatchNorm3d should only accept 5D input."""
        bn = BatchNorm3d(num_features=8)

        # Should work with 5D
        bn(nova.randn(2, 8, 4, 4, 4))

        # Should fail with other dimensions
        with pytest.raises(ValueError, match="expected 5D input"):
            bn(nova.randn(2, 8, 4, 4))  # 4D


class TestBatchNormEdgeCases:
    """Test edge cases and special configurations."""

    def test_track_running_stats_false(self):
        """Test behavior when track_running_stats=False."""
        bn = BatchNorm1d(num_features=10, track_running_stats=False)

        # Should not have running stats
        assert bn.running_mean is None
        assert bn.running_var is None
        assert bn.num_batches_tracked is None

        # Should use batch stats in both training and eval mode
        bn.train()
        x = nova.randn(32, 10)
        y_train = bn(x)
        assert y_train.shape == x.shape

        bn.eval()
        y_eval = bn(x)
        assert y_eval.shape == x.shape

    def test_momentum_none_uses_cumulative_average(self):
        """Test that momentum=None uses cumulative moving average."""
        bn = BatchNorm1d(num_features=5, momentum=None)
        bn.train()

        # First batch
        x1 = nova.ones((10, 5)) * 10.0
        _ = bn(x1)

        # num_batches_tracked should be 1
        assert bn.num_batches_tracked.item() == 1

        # Second batch
        x2 = nova.ones((10, 5)) * 20.0
        _ = bn(x2)

        # num_batches_tracked should be 2
        assert bn.num_batches_tracked.item() == 2

    def test_reset_parameters(self):
        """Test that reset_parameters properly initializes everything."""
        bn = BatchNorm2d(num_features=16)

        init.constant_(bn.weight, 5.0)
        init.constant_(bn.bias, 3.0)
        init.constant_(bn.running_mean, 10.0)
        init.constant_(bn.running_var, 0.5)

        # Reset
        bn.reset_parameters()

        # Check reset values
        assert nova.allclose(bn.weight, nova.ones((16,)))
        assert nova.allclose(bn.bias, nova.zeros((16,)))
        assert nova.allclose(bn.running_mean, nova.zeros((16,)))
        assert nova.allclose(bn.running_var, nova.ones((16,)))
