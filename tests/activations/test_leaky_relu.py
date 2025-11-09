import pytest
import numpy as np

from src.layers.activations.relu import LeakyReLU
from src.utils import numeric_grad_elementwise

RNG = np.random.RandomState(0)

ATOL = 1e-6
RTOL = 1e-5


def test_leaky_relu_forward_backward_and_numeric():
    X = RNG.randn(6, 4)
    slope = 0.1
    act = LeakyReLU(negative_slope=slope)

    Y = act.forward(X)
    assert Y.shape == X.shape
    assert np.allclose(Y[X > 0], X[X > 0])
    assert np.allclose(Y[X < 0], X[X < 0] * slope)

    act.forward(X)
    back = act.backward(np.ones_like(X))
    expected = np.where(X > 0, 1, slope)
    assert np.allclose(back, expected, atol=ATOL, rtol=RTOL)

    numg = numeric_grad_elementwise(lambda z: act.forward(z), X.copy(), eps=1e-6)
    mask = X != 0
    assert np.allclose(numg[mask], back[mask], atol=ATOL, rtol=RTOL)
