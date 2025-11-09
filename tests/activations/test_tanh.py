import pytest
import numpy as np

from src.layers.activations.tanh import Tanh
from src.utils import numeric_grad_elementwise

RNG = np.random.RandomState(0)

ATOL = 1e-6
RTOL = 1e-5


def test_tanh_forward_backward_and_numeric():
    X = RNG.randn(5, 5)
    act = Tanh()

    Y = act.forward(X)
    assert Y.shape == X.shape
    assert np.all(Y > -1) and np.all(Y < 1)
    # oddness: tanh(-x) == -tanh(x)
    assert np.allclose(act.forward(-X), -Y, atol=1e-7)

    numg = numeric_grad_elementwise(lambda z: act.forward(z), X.copy(), eps=1e-6)
    act.forward(X)
    back = act.backward(np.ones_like(X))
    expected = 1 - Y**2
    assert np.allclose(numg, expected, atol=1e-6, rtol=1e-6)
    assert np.allclose(back, expected, atol=1e-6, rtol=1e-6)
