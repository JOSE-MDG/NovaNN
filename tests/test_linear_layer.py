# tests/test_layers_minimal.py
import numpy as np
import pytest

from novann.layers import Linear
from novann.utils import numeric_grad_scalar_wrt_x, numeric_grad_wrt_param


RNG = np.random.RandomState(0)


def test_linear_forward_shape_and_numeric_backward():
    # Setup: small linear layer with deterministic weight initializer
    in_f, out_f, B = 3, 2, 5
    init_fn = lambda shape: np.ones(shape, dtype=float)
    lin = Linear(in_features=in_f, out_features=out_f, bias=True, init=init_fn)

    # Input: batch of random vectors
    X = RNG.randn(B, in_f)

    # Forward: shape check and simple numeric expectation (weights are ones)
    out = lin.forward(X)
    assert out.shape == (B, out_f)
    expected = X @ np.ones((out_f, in_f)).T
    assert np.allclose(out, expected)

    # Backward: analytic gradient w.r.t. input vs numerical finite differences
    G = RNG.randn(B, out_f)
    lin.forward(X)  # ensure cache is set
    dx = lin.backward(G)
    num_dx = numeric_grad_scalar_wrt_x(lambda z: lin.forward(z), X.copy(), G, eps=1e-6)
    assert np.allclose(dx, num_dx, atol=1e-5, rtol=1e-5)


def test_linear_param_grads_numeric():
    # Setup: random (but deterministic) initialization for numeric gradient checks
    in_f, out_f, B = 4, 3, 6
    init_fn = lambda shape: RNG.randn(*shape)
    lin = Linear(in_features=in_f, out_features=out_f, bias=True, init=init_fn)

    X = RNG.randn(B, in_f)
    G = RNG.randn(B, out_f)

    # Ensure forward caches are available and compute analytic gradients
    lin.forward(X)
    _ = lin.backward(G)
    analytic_w_grad = lin.weight.grad.copy()
    analytic_b_grad = lin.bias.grad.copy()

    # Numeric gradients by perturbing parameters directly
    num_w_grad = numeric_grad_wrt_param(lin, "weight", X.copy(), G, eps=1e-6)
    num_b_grad = numeric_grad_wrt_param(lin, "bias", X.copy(), G, eps=1e-6)

    # Compare analytic vs numeric parameter gradients
    assert np.allclose(analytic_w_grad, num_w_grad, atol=1e-6, rtol=1e-5)
    assert np.allclose(analytic_b_grad, num_b_grad, atol=1e-6, rtol=1e-5)
