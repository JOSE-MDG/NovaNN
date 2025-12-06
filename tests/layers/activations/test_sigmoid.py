import pytest
import numpy as np

from novann.layers import Sigmoid
from novann.utils.gradient_checking import numeric_grad_elementwise

RNG = np.random.RandomState(0)


def test_sigmoid_forward_backward_and_numeric():
    # random inputs
    X = RNG.randn(5, 5)
    act = Sigmoid()

    # Forward: shape and range (0,1)
    Y = act(X)
    assert Y.shape == X.shape
    assert np.all(Y > 0) and np.all(Y < 1)

    # Numeric gradient (finite differences)
    numg = numeric_grad_elementwise(lambda z: act(z), X.copy(), eps=1e-6)

    # Analytical gradient via backward (uses cached forward output)
    act(X)
    back = act.backward(np.ones_like(X))  # upstream ones -> d sum(sigmoid) / dx

    expected = Y * (1 - Y)
    assert np.allclose(numg, expected, atol=1e-6, rtol=1e-6)
    assert np.allclose(back, expected, atol=1e-6, rtol=1e-6)
