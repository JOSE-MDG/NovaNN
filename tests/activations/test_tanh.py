import pytest
import numpy as np

from novann.layers import Tanh
from novann.utils.gradient_checking import numeric_grad_elementwise

RNG = np.random.RandomState(0)

ATOL = 1e-6
RTOL = 1e-5


def test_tanh_forward_backward_and_numeric():
    # Random inputs
    X = RNG.randn(5, 5)
    act = Tanh()

    # Forward: shape and range (-1, 1)
    Y = act.forward(X)
    assert Y.shape == X.shape
    assert np.all(Y > -1) and np.all(Y < 1)

    # Property: odd function -> tanh(-x) == -tanh(x)
    assert np.allclose(act.forward(-X), -Y, atol=1e-7)

    # Numeric gradient (finite differences)
    numg = numeric_grad_elementwise(lambda z: act.forward(z), X.copy(), eps=1e-6)

    # Analytical gradient via backward (uses cached forward output)
    act.forward(X)
    back = act.backward(np.ones_like(X))

    expected = 1 - Y**2  # derivative of tanh
    assert np.allclose(numg, expected, atol=ATOL, rtol=RTOL)
    assert np.allclose(back, expected, atol=ATOL, rtol=RTOL)
