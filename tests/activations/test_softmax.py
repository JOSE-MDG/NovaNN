import numpy as np
import pytest

from novann.layers import SoftMax
from novann.utils.gradient_checking import numeric_grad_scalar_from_softmax

RNG = np.random.RandomState(0)


def test_softmax_forward_properties_and_shift_invariance_columnwise():
    X = RNG.randn(10, 6) * 30.0
    act = SoftMax()

    Y = act(X)
    sums = Y.sum(axis=1)
    # probabilities per row sum to one
    assert np.allclose(sums, np.ones_like(sums), atol=1e-7)
    # non-negativity of softmax outputs
    assert np.all(Y >= 0)

    # shift invariance: adding a constant to logits should not change softmax
    c = 5.0
    Y_shift = act(X + c)
    assert np.allclose(Y, Y_shift, atol=1e-7)


def test_softmax_backward_numeric_columnwise():
    X = RNG.randn(7, 5)
    act = SoftMax()
    G = RNG.randn(*X.shape)

    # numerical Jacobian-vector product for verification
    num_grad = numeric_grad_scalar_from_softmax(
        lambda z: act.forward(z), X.copy(), G, eps=1e-6
    )

    act(X)
    back = act.backward(G)

    # compare analytic Jv product to numerical approximation
    assert np.allclose(num_grad, back, atol=1e-5, rtol=1e-5)
