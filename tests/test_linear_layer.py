# tests/test_layers_minimal.py
import numpy as np
import pytest

from src.layers.linear.linear import Linear
from src.layers.linear.linear import Linear
from src.utils import numeric_grad_scalar_wrt_x, numeric_grad_wrt_param


RNG = np.random.RandomState(0)


def test_linear_forward_shape_and_numeric_backward():
    in_f, out_f, B = 3, 2, 5
    # init: weight filled with ones so forward is easy to reason
    init_fn = lambda shape: np.ones(shape, dtype=float)
    lin = Linear(in_features=in_f, out_features=out_f, bias=True, init=init_fn)

    # input shape: (batch,in_features)
    X = RNG.randn(B, in_f)

    out = lin.forward(X)
    assert out.shape == (B, out_f)

    # default bias zeros; since W is ones, out should equal row-wise sum of X
    expected = X @ np.ones((out_f, in_f)).T  # (B,in_f) @ (in_f,out) -> (B,out)
    assert np.allclose(out, expected)

    # Numeric check of backward wrt input:
    # choose arbitrary upstream gradient G
    G = RNG.randn(B, out_f)
    # run forward again to set cache inside layer
    lin.forward(X)
    dx = lin.backward(G)  # analytic dx from layer

    # numeric gradient w.r.t. input
    num_dx = numeric_grad_scalar_wrt_x(lambda z: lin.forward(z), X.copy(), G, eps=1e-6)
    assert np.allclose(dx, num_dx, atol=1e-5, rtol=1e-5)


def test_linear_param_grads_numeric():
    in_f, out_f, B = 4, 3, 6
    # random init but deterministic
    init_fn = lambda shape: RNG.randn(*shape)
    lin = Linear(in_features=in_f, out_features=out_f, bias=True, init=init_fn)

    X = RNG.randn(B, in_f)
    G = RNG.randn(B, out_f)

    # ensure forward caches are set
    lin.forward(X)
    # analytic backward will fill lin.weight.grad and lin.bias.grad
    _ = lin.backward(G)
    analytic_w_grad = lin.weight.grad.copy()
    analytic_b_grad = lin.bias.grad.copy()

    # numeric grads by perturbing parameters
    num_w_grad = numeric_grad_wrt_param(lin, "weight", X.copy(), G, eps=1e-6)
    num_b_grad = numeric_grad_wrt_param(lin, "bias", X.copy(), G, eps=1e-6)

    assert np.allclose(analytic_w_grad, num_w_grad, atol=1e-6, rtol=1e-5)
    assert np.allclose(analytic_b_grad, num_b_grad, atol=1e-6, rtol=1e-5)
