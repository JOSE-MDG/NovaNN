import numpy as np
import pytest

from src.model.nn import Sequential
from src.layers.linear.linear import Linear
from src.layers.activations.tanh import Tanh

RNG = np.random.RandomState(0)


def test_sequential_forward_backward_shape_and_parameters_collection():
    in_f, hidden, out_f, B = 3, 4, 2, 5
    l1 = Linear(in_features=in_f, out_features=hidden, bias=True)
    act = Tanh()
    l2 = Linear(in_features=hidden, out_features=out_f, bias=True)

    seq = Sequential(l1, act, l2)  # Automatically detects the activation function Tanh
    seq.train()

    X = RNG.randn(in_f, B)
    out = seq.forward(X)
    assert out.shape == (out_f, B)

    G = RNG.randn(out_f, B)
    dx = seq.backward(G)
    assert dx.shape == X.shape

    params = list(seq.parameters())
    assert len(params) == 4
    for p in params:
        assert hasattr(p, "data") and hasattr(p, "grad")
        assert getattr(p, "name", None) is None
