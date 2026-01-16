import nova
import pytest
from nova.nn import (
    LazyBatchNorm1d,
    LazyBatchNorm2d,
    LazyBatchNorm3d,
    LazyConv1d,
    LazyConv2d,
    LazyConv3d,
    LazyLinear,
)

nova.manual_seed(8)


# -- BatchNorm --


class TestLazyBatchNorm:
    """Test lazy initialization variants."""

    def test_lazy_batchnorm1d_infers_num_features(self):
        """LazyBatchNorm1d should infer num_features from input."""
        bn = LazyBatchNorm1d()

        # Before first forward, parameters are uninitialized
        assert bn.has_uninitialized_params()

        # First forward pass with 20 features
        x = nova.randn(8, 20)
        y = bn(x)

        # Should infer num_features = 20
        assert bn.num_features == 20
        assert not bn.has_uninitialized_params()
        assert y.shape == x.shape

        # Subsequent forwards should work normally
        x2 = nova.randn(4, 20)
        y2 = bn(x2)
        assert y2.shape == x2.shape

    def test_lazy_batchnorm2d_infers_channels(self):
        """LazyBatchNorm2d should infer num_features from channel dimension."""
        bn = LazyBatchNorm2d()

        x = nova.randn(4, 32, 16, 16)
        y = bn(x)

        assert bn.num_features == 32
        assert y.shape == x.shape

    def test_lazy_batchnorm3d_infers_channels(self):
        """LazyBatchNorm3d should infer num_features from channel dimension."""
        bn = LazyBatchNorm3d()

        x = nova.randn(2, 16, 8, 8, 8)
        y = bn(x)

        assert bn.num_features == 16
        assert y.shape == x.shape


# -- Conv --


class TestLazyConv1d:
    def test_infers_in_channels(self):
        m = LazyConv1d(out_channels=16, kernel_size=3, padding=1)
        assert m.has_uninitialized_params()

        x = nova.randn(2, 8, 10)
        y = m(x)

        assert m.in_channels == 8
        assert not m.has_uninitialized_params()
        assert y.shape == (2, 16, 10)

    def test_subsequent_forward_works(self):
        m = LazyConv1d(16, kernel_size=3, padding=1)
        x1 = nova.randn(2, 5, 10)
        _ = m(x1)

        x2 = nova.randn(1, 5, 8)
        y = m(x2)
        assert y.shape == (1, 16, 8)


class TestLazyConv2d:
    def test_infers_in_channels(self):
        m = LazyConv2d(out_channels=32, kernel_size=3, padding=1)
        assert m.has_uninitialized_params()

        x = nova.randn(2, 16, 8, 8)
        y = m(x)

        assert m.in_channels == 16
        assert not m.has_uninitialized_params()
        assert y.shape == (2, 32, 8, 8)

    def test_subsequent_forward_works(self):
        m = LazyConv2d(32, kernel_size=3, padding=1)
        x1 = nova.randn(1, 3, 8, 8)
        _ = m(x1)

        x2 = nova.randn(2, 3, 16, 16)
        y = m(x2)
        assert y.shape == (2, 32, 16, 16)


class TestLazyConv3d:
    def test_infers_in_channels(self):
        m = LazyConv3d(out_channels=16, kernel_size=3, padding=1)
        assert m.has_uninitialized_params()

        x = nova.randn(1, 8, 4, 4, 4)
        y = m(x)

        assert m.in_channels == 8
        assert not m.has_uninitialized_params()
        assert y.shape == (1, 16, 4, 4, 4)

    def test_subsequent_forward_works(self):
        m = LazyConv3d(16, kernel_size=3, padding=1)
        x1 = nova.randn(1, 3, 4, 4, 4)
        _ = m(x1)

        x2 = nova.randn(2, 3, 8, 8, 8)
        y = m(x2)
        assert y.shape == (2, 16, 8, 8, 8)


# -- Linear --


class TestLazyLinear:
    def test_infers_in_features(self):
        m = LazyLinear(out_features=30)
        assert m.has_uninitialized_params()

        x = nova.randn(128, 20)
        y = m(x)

        assert m.in_features == 20
        assert not m.has_uninitialized_params()
        assert y.shape == (128, 30)

    def test_subsequent_forward_works(self):
        m = LazyLinear(30)
        x1 = nova.randn(64, 15)
        _ = m(x1)

        x2 = nova.randn(32, 15)
        y = m(x2)
        assert y.shape == (32, 30)

    def test_multidimensional_input(self):
        m = LazyLinear(50)
        x = nova.randn(10, 5, 20)
        y = m(x)
        assert m.in_features == 20
        assert y.shape == (10, 5, 50)
