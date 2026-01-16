import pytest
import nova
from nova.nn.modules import Linear
from nova.utils.grad_checking import grad_check_wrt_inputs

nova.manual_seed(8)


class TestLinear:
    def test_forward_shape(self):
        m = Linear(20, 30)
        x = nova.randn(128, 20)
        y = m(x)
        assert y.shape == (128, 30)

    def test_backward_gradients(self):
        m = Linear(10, 5)
        x = nova.randn(4, 10, requires_grad=True)

        def forward(inp):
            return m(inp).sum()

        analytic, numeric = grad_check_wrt_inputs(forward, x, eps=1e-4)
        assert nova.allclose(analytic[0], numeric[0], rtol=5e-3, atol=5e-3)

    def test_without_bias(self):
        m = Linear(20, 30, bias=False)
        assert m.bias is None
        x = nova.randn(128, 20)
        y = m(x)
        assert y.shape == (128, 30)

    def test_multidimensional_input(self):
        m = Linear(20, 30)
        x = nova.randn(10, 5, 20)
        y = m(x)
        assert y.shape == (10, 5, 30)
