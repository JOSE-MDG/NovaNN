import pytest
import nova
from nova.utils import grad_check_wrt_inputs
from nova.nn import (
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    GlobalAvgPool1d,
    GlobalAvgPool2d,
    GlobalAvgPool3d,
)

nova.manual_seed(8)


class TestMaxPool1d:
    def test_forward_shape(self):
        pool = MaxPool1d(kernel_size=2, stride=2)
        x = nova.randn(4, 16, 100)
        y = pool(x)
        assert y.shape == (4, 16, 50)

    def test_backward_gradients(self):
        pool = MaxPool1d(kernel_size=2, stride=2)
        x = nova.randn(2, 8, 20, requires_grad=True)

        forward = lambda inp: pool(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)

    def test_with_padding(self):
        pool = MaxPool1d(kernel_size=3, stride=1, padding=1)
        x = nova.randn(2, 32, 50)
        y = pool(x)
        assert y.shape == (2, 32, 50)

    def test_with_dilation(self):
        pool = MaxPool1d(kernel_size=3, stride=1, dilation=2)
        x = nova.randn(4, 16, 50)
        y = pool(x)
        assert y.shape == (4, 16, 46)


class TestMaxPool2d:
    def test_forward_shape(self):
        pool = MaxPool2d(kernel_size=2, stride=2)
        x = nova.randn(32, 64, 56, 56)
        y = pool(x)
        assert y.shape == (32, 64, 28, 28)

    def test_backward_gradients(self):
        pool = MaxPool2d(kernel_size=2, stride=2)
        x = nova.randn(2, 4, 8, 8, requires_grad=True)

        forward = lambda inp: pool(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)

    def test_non_square_kernel(self):
        pool = MaxPool2d(kernel_size=(3, 2), stride=(3, 2))
        x = nova.randn(16, 128, 32, 32)
        y = pool(x)
        assert y.shape == (16, 128, 10, 16)


class TestMaxPool3d:
    def test_forward_shape(self):
        pool = MaxPool3d(kernel_size=2, stride=2)
        x = nova.randn(8, 64, 16, 112, 112)
        y = pool(x)
        assert y.shape == (8, 64, 8, 56, 56)

    def test_backward_gradients(self):
        pool = MaxPool3d(kernel_size=2, stride=2)
        x = nova.randn(1, 4, 4, 4, 4, requires_grad=True)

        forward = lambda inp: pool(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)

    def test_preserve_temporal_dimension(self):
        pool = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = nova.randn(4, 128, 8, 56, 56)
        y = pool(x)
        assert y.shape == (4, 128, 8, 28, 28)


class TestAvgPool1d:
    def test_forward_shape(self):
        pool = AvgPool1d(kernel_size=2, stride=2)
        x = nova.randn(4, 16, 100)
        y = pool(x)
        assert y.shape == (4, 16, 50)

    def test_backward_gradients(self):
        pool = AvgPool1d(kernel_size=2, stride=2)
        x = nova.randn(2, 8, 20, requires_grad=True)

        forward = lambda inp: pool(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)


class TestAvgPool2d:
    def test_forward_shape(self):
        pool = AvgPool2d(kernel_size=2, stride=2)
        x = nova.randn(16, 64, 32, 32)
        y = pool(x)
        assert y.shape == (16, 64, 16, 16)

    def test_backward_gradients(self):
        pool = AvgPool2d(kernel_size=2, stride=2)
        x = nova.randn(2, 4, 8, 8, requires_grad=True)

        forward = lambda inp: pool(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)

    def test_with_padding(self):
        pool = AvgPool2d(kernel_size=3, stride=1, padding=1)
        x = nova.randn(4, 16, 8, 8)
        y = pool(x)
        assert y.shape == (4, 16, 8, 8)


class TestAvgPool3d:
    def test_forward_shape(self):
        pool = AvgPool3d(kernel_size=2, stride=2)
        x = nova.randn(4, 3, 16, 112, 112)
        y = pool(x)
        assert y.shape == (4, 3, 8, 56, 56)

    def test_backward_gradients(self):
        pool = AvgPool3d(kernel_size=2, stride=2)
        x = nova.randn(1, 4, 4, 4, 4, requires_grad=True)

        forward = lambda inp: pool(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)


class TestGlobalAvgPool1d:
    def test_forward_shape(self):
        pool = GlobalAvgPool1d()
        x = nova.randn(8, 64, 100)
        y = pool(x)
        assert y.shape == (8, 64, 1)

    def test_backward_gradients(self):
        pool = GlobalAvgPool1d()
        x = nova.randn(4, 16, 50, requires_grad=True)

        forward = lambda inp: pool(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)


class TestGlobalAvgPool2d:
    def test_forward_shape(self):
        pool = GlobalAvgPool2d()
        x = nova.randn(32, 512, 7, 7)
        y = pool(x)
        assert y.shape == (32, 512, 1, 1)

    def test_backward_gradients(self):
        pool = GlobalAvgPool2d()
        x = nova.randn(4, 16, 8, 8, requires_grad=True)

        forward = lambda inp: pool(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)


class TestGlobalAvgPool3d:
    def test_forward_shape(self):
        pool = GlobalAvgPool3d()
        x = nova.randn(8, 512, 8, 7, 7)
        y = pool(x)
        assert y.shape == (8, 512, 1, 1, 1)

    def test_backward_gradients(self):
        pool = GlobalAvgPool3d()
        x = nova.randn(2, 8, 4, 4, 4, requires_grad=True)

        forward = lambda inp: pool(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)
