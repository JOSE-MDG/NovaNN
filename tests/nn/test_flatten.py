import pytest
import nova
from nova.utils import grad_check_wrt_inputs
from nova.nn import Flatten

nova.manual_seed(8)


class TestFlatten:
    def test_default_flatten(self):
        """Test default behavior: flatten all except batch dimension."""
        flatten = Flatten()
        x = nova.randn(32, 3, 28, 28)
        y = flatten(x)
        assert y.shape == (32, 2352)  # 3 * 28 * 28 = 2352

    def test_backward_gradients(self):
        """Test gradient computation through flatten."""
        flatten = Flatten()
        x = nova.randn(4, 8, 6, 6, requires_grad=True)

        forward = lambda inp: flatten(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=1e-2, atol=1e-2)

    def test_custom_range(self):
        """Test flattening custom dimension range."""
        flatten = Flatten(start_dim=2, end_dim=3)
        x = nova.randn(10, 5, 3, 4, 7)
        y = flatten(x)
        assert y.shape == (10, 5, 12, 7)  # 3 * 4 = 12

    def test_flatten_all(self):
        """Test flattening including batch dimension."""
        flatten = Flatten(start_dim=0)
        x = nova.randn(2, 3, 4, 5)
        y = flatten(x)
        assert y.shape == (120,)  # 2 * 3 * 4 * 5 = 120

    def test_single_dimension(self):
        """Test that flattening single dimension works."""
        flatten = Flatten(start_dim=1, end_dim=1)
        x = nova.randn(8, 16, 10)
        y = flatten(x)
        assert y.shape == (8, 16, 10)  # No change
