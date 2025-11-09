import numpy as np
import pytest

from src.layers.activations.softmax import SoftMax
from src.utils import numeric_grad_scalar_from_softmax

RNG = np.random.RandomState(0)


def test_softmax_forward_properties_and_shift_invariance_columnwise():
    X = RNG.randn(6, 10) * 30.0
    act = SoftMax()

    Y = act.forward(X)
    sums = Y.sum(axis=0)
    assert np.allclose(sums, np.ones_like(sums), atol=1e-7)
    assert np.all(Y >= 0)

    c = 5.0
    Y_shift = act.forward(X + c)
    assert np.allclose(Y, Y_shift, atol=1e-7)


def test_softmax_backward_numeric_columnwise():
    X = RNG.randn(5, 7)
    act = SoftMax()
    G = RNG.randn(*X.shape)

    num_grad = numeric_grad_scalar_from_softmax(
        lambda z: act.forward(z), X.copy(), G, eps=1e-6
    )

    act.forward(X)
    back = act.backward(G)

    assert np.allclose(num_grad, back, atol=1e-5, rtol=1e-5)
