import pytest
import nova
from nova.utils import grad_check_wrt_inputs
from nova.nn import Dropout, Dropout2d, Dropout3d

nova.manual_seed(8)


class TestDropout:
    def test_forward_shape(self):
        """Test that dropout preserves shape."""
        dropout = Dropout(p=0.5)
        dropout.train()
        x = nova.randn(32, 128)
        y = dropout(x)
        assert y.shape == (32, 128)

    def test_eval_mode_no_dropout(self):
        """Test that dropout is disabled in eval mode."""
        dropout = Dropout(p=0.5)
        dropout.eval()
        x = nova.randn(16, 64)
        y = dropout(x)
        assert nova.allclose(x, y)

    def test_train_mode_applies_dropout(self):
        """Test that dropout is applied in train mode."""
        dropout = Dropout(p=0.5)
        dropout.train()
        x = nova.ones((100, 100))
        y = dropout(x)
        # Should have some zeros
        assert (y == 0).sum() > 0

    def test_zero_probability(self):
        """Test that p=0 means no dropout."""
        dropout = Dropout(p=0.0)
        dropout.train()
        x = nova.randn(10, 20)
        y = dropout(x)
        assert nova.allclose(x, y)


class TestDropout2d:
    def test_forward_shape(self):
        """Test that dropout2d preserves shape."""
        dropout = Dropout2d(p=0.2)
        dropout.train()
        x = nova.randn(8, 64, 32, 32)
        y = dropout(x)
        assert y.shape == (8, 64, 32, 32)

    def test_eval_mode_no_dropout(self):
        """Test that dropout2d is disabled in eval mode."""
        dropout = Dropout2d(p=0.5)
        dropout.eval()
        x = nova.randn(4, 32, 16, 16)
        y = dropout(x)
        assert nova.allclose(x, y)

    def test_drops_entire_channels(self):
        """Test that entire channels are dropped together."""
        dropout = Dropout2d(p=0.5)
        dropout.train()
        x = nova.ones((1, 10, 8, 8))
        y = dropout(x)

        # Check that for each channel, either all values are 0 or all are scaled
        for c in range(10):
            channel = y[0, c, :, :]
            unique_vals = nova.unique(channel)
            # Should have at most 2 unique values (0 and scaled value)
            assert len(unique_vals) <= 2

    def test_backward_gradients(self):
        """Test gradient computation through dropout2d."""
        dropout = Dropout2d(p=0.3)
        dropout.train()
        x = nova.randn(2, 8, 4, 4, requires_grad=True)

        forward = lambda inp: dropout(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=1e-2, atol=1e-2)


class TestDropout3d:
    def test_forward_shape(self):
        """Test that dropout3d preserves shape."""
        dropout = Dropout3d(p=0.3)
        dropout.train()
        x = nova.randn(4, 32, 16, 64, 64)
        y = dropout(x)
        assert y.shape == (4, 32, 16, 64, 64)

    def test_eval_mode_no_dropout(self):
        """Test that dropout3d is disabled in eval mode."""
        dropout = Dropout3d(p=0.5)
        dropout.eval()
        x = nova.randn(2, 16, 8, 8, 8)
        y = dropout(x)
        assert nova.allclose(x, y)

    def test_drops_entire_channels(self):
        """Test that entire 3D channels are dropped together."""
        dropout = Dropout3d(p=0.5)
        dropout.train()
        x = nova.ones((1, 8, 4, 4, 4))
        y = dropout(x)

        # Check that for each channel, either all values are 0 or all are scaled
        for c in range(8):
            channel = y[0, c, :, :, :]
            unique_vals = nova.unique(channel)
            # Should have at most 2 unique values (0 and scaled value)
            assert len(unique_vals) <= 2

    def test_backward_gradients(self):
        """Test gradient computation through dropout3d."""
        dropout = Dropout3d(p=0.3)
        dropout.train()
        x = nova.randn(1, 4, 4, 4, 4, requires_grad=True)

        forward = lambda inp: dropout(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=1e-2, atol=1e-2)
