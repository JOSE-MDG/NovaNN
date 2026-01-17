import pytest
import nova
from nova.utils import grad_check_wrt_inputs
from nova.nn import Conv1d, Conv2d, Conv3d

nova.manual_seed(8)


class TestConv1d:
    def test_forward_shape(self):
        m = Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = nova.randn(4, 3, 10)
        y = m(x)
        assert y.shape == (4, 16, 10)

    def test_backward_gradients(self):
        m = Conv1d(in_channels=2, out_channels=3, kernel_size=3)
        x = nova.randn(2, 2, 8, requires_grad=True)

        def forward(inp):
            return m(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)

    def test_without_bias(self):
        m = Conv1d(3, 16, kernel_size=3, bias=False)
        assert m.bias is None
        x = nova.randn(2, 3, 10)
        y = m(x)
        assert y.shape == (2, 16, 8)

    def test_stride_and_padding(self):
        m = Conv1d(3, 8, kernel_size=3, stride=2, padding=1)
        x = nova.randn(2, 3, 10)
        y = m(x)
        assert y.shape == (2, 8, 5)


class TestConv2d:
    def test_forward_shape(self):
        m = Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        x = nova.randn(2, 3, 8, 8)
        y = m(x)
        assert y.shape == (2, 64, 8, 8)

    def test_backward_gradients(self):
        m = Conv2d(in_channels=2, out_channels=4, kernel_size=3)
        x = nova.randn(1, 2, 6, 6, requires_grad=True)

        def forward(inp):
            return m(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)

    def test_without_bias(self):
        m = Conv2d(3, 32, kernel_size=3, bias=False)
        assert m.bias is None
        x = nova.randn(1, 3, 8, 8)
        y = m(x)
        assert y.shape == (1, 32, 6, 6)

    def test_stride_and_padding(self):
        m = Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        x = nova.randn(1, 3, 8, 8)
        y = m(x)
        assert y.shape == (1, 16, 4, 4)

    def test_asymmetric_kernel(self):
        m = Conv2d(3, 8, kernel_size=(3, 5), padding=(1, 2))
        x = nova.randn(1, 3, 8, 8)
        y = m(x)
        assert y.shape == (1, 8, 8, 8)


class TestConv3d:
    def test_forward_shape(self):
        m = Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = nova.randn(1, 3, 4, 8, 8)
        y = m(x)
        assert y.shape == (1, 16, 4, 8, 8)

    def test_backward_gradients(self):
        m = Conv3d(in_channels=2, out_channels=3, kernel_size=3)
        x = nova.randn(1, 2, 4, 4, 4, requires_grad=True)

        def forward(inp):
            return m(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-2, atol=5e-2)

    def test_without_bias(self):
        m = Conv3d(3, 8, kernel_size=3, bias=False)
        assert m.bias is None
        x = nova.randn(1, 3, 4, 4, 4)
        y = m(x)
        assert y.shape == (1, 8, 2, 2, 2)

    def test_stride_and_padding(self):
        m = Conv3d(3, 8, kernel_size=3, stride=2, padding=1)
        x = nova.randn(1, 3, 8, 8, 8)
        y = m(x)
        assert y.shape == (1, 8, 4, 4, 4)
