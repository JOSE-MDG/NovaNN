import pytest
import numpy as np

from src.layers.activations.relu import ReLU
from src.utils import numeric_grad_elementwise

RNG = np.random.RandomState(0)  # deterministic RNG for reproducible tests

ATOL = 1e-6
RTOL = 1e-5


def test_relu_forward_backward_and_numeric():
    # Create random input batch
    X = RNG.randn(6, 4)
    act = ReLU()

    # Forward: shape and non-negativity property
    Y = act.forward(X)
    assert Y.shape == X.shape
    assert np.all(Y >= 0)

    # Backward: analytical gradient equals indicator(X > 0)
    act.forward(X)
    back = act.backward(np.ones_like(X))
    expected = (X > 0).astype(float)
    assert np.array_equal(back, expected)

    # Numeric gradient (finite differences) for non-zero inputs
    numg = numeric_grad_elementwise(lambda z: act.forward(z), X.copy(), eps=1e-6)
    mask = X != 0  # exclude points where derivative is undefined (x == 0)
    assert np.allclose(numg[mask], back[mask], atol=ATOL, rtol=RTOL)
