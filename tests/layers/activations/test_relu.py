import pytest
import numpy as np

from novann.layers import ReLU
from novann.utils.gradient_checking import numeric_grad_elementwise

RNG = np.random.RandomState(0)  # Deterministic RNG for reproducible tests


def test_relu_forward_backward_and_numeric():
    """Test forward and backward passes of ReLU layer with numerical validation."""
    # Create random input batch
    X = RNG.randn(6, 4)
    act = ReLU()

    # Forward: shape and non-negativity property
    Y = act(X)
    assert Y.shape == X.shape
    assert np.all(Y >= 0)

    # Backward: analytical gradient equals indicator(X > 0)
    act(X)
    back = act.backward(np.ones_like(X))
    expected = (X > 0).astype(float)
    assert np.array_equal(back, expected)

    # Numeric gradient (finite differences) for non-zero inputs
    numg = numeric_grad_elementwise(lambda z: act(z), X.copy(), eps=1e-6)
    mask = X != 0  # exclude points where derivative is undefined (x == 0)
    assert np.allclose(numg[mask], back[mask], atol=1e-6, rtol=1e-5)
