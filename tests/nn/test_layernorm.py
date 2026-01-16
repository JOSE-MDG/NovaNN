import pytest
import nova
from nova.utils import grad_check_wrt_inputs
from nova.nn import LayerNorm

nova.manual_seed(8)


class TestLayerNorm:
    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        ln = LayerNorm(512)
        x = nova.randn(32, 10, 512)
        y = ln(x)
        assert y.shape == (32, 10, 512)

    def test_normalization_properties(self):
        """Test that normalization produces mean~0 and var~1."""
        ln = LayerNorm(64)
        x = nova.randn(8, 64)
        y = ln(x)

        # Check mean is close to 0 for each sample
        mean = y.mean(dim=1)
        assert nova.allclose(mean, nova.zeros_like(mean), atol=1e-5)

        # Check variance is close to 1 for each sample
        var = y.var(dim=1)
        assert nova.allclose(var, nova.ones_like(var), atol=1e-4)

    def test_backward_gradients(self):
        """Test gradient computation through LayerNorm."""
        ln = LayerNorm(128)
        x = nova.randn(4, 128, requires_grad=True)

        forward = lambda inp: ln(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=0.1, atol=0.1)

    def test_without_affine(self):
        """Test LayerNorm without learnable parameters."""
        ln = LayerNorm(64, elementwise_affine=False)
        assert ln.weight is None
        assert ln.bias is None

        x = nova.randn(4, 64)
        y = ln(x)

        # Should still normalize even without affine parameters
        mean = y.mean(dim=1)
        assert nova.allclose(mean, nova.zeros_like(mean), atol=1e-5)

    def test_multidimensional_normalization(self):
        """Test normalization over multiple dimensions."""
        ln = LayerNorm((3, 5))
        x = nova.randn(20, 3, 5)
        y = ln(x)

        assert y.shape == (20, 3, 5)

        # Check normalization over last 2 dimensions
        y_flat = y.reshape(20, -1)
        mean = y_flat.mean(dim=1)
        assert nova.allclose(mean, nova.zeros_like(mean), atol=1e-5)
