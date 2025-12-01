import pytest
import numpy as np

from novann.layers.activations.relu import LeakyReLU
from novann.utils import numeric_grad_elementwise

RNG = np.random.RandomState(0)  # Deterministic RNG for reproducible tests

ATOL = 1e-6
RTOL = 1e-5


def test_leaky_relu_forward_backward_and_numeric():
    # Random input batch
    X = RNG.randn(6, 4)
    slope = 0.1  # negative slope for LeakyReLU
    act = LeakyReLU(negative_slope=slope)

    # Forward: shape and element-wise behaviour
    Y = act.forward(X)
    assert Y.shape == X.shape
    assert np.allclose(Y[X > 0], X[X > 0])
    assert np.allclose(Y[X < 0], X[X < 0] * slope)

    # Backward: analytical gradient check
    act.forward(X)
    back = act.backward(np.ones_like(X))
    expected = np.where(X > 0, 1, slope)
    assert np.allclose(back, expected, atol=ATOL, rtol=RTOL)

    # Numeric gradient (finite differences) for non-zero inputs
    numg = numeric_grad_elementwise(lambda z: act.forward(z), X.copy(), eps=1e-6)
    mask = X != 0
    assert np.allclose(numg[mask], back[mask], atol=ATOL, rtol=RTOL)
