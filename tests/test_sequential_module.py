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

    # Sequential should detect activation placement and apply sensible initializers
    seq = Sequential(l1, act, l2)
    seq.train()

    # Internal helper returns the init key for the activation after layer 0
    activation, _ = seq._find_next_activation(0)
    assert activation == "tanh"  # expect lowercased class name used as key

    X = RNG.randn(B, in_f)
    out = seq.forward(X)
    assert out.shape == (B, out_f)  # output shape matches batch and out_features

    G = RNG.randn(B, out_f)
    dx = seq.backward(G)
    assert dx.shape == X.shape  # backward returns gradient w.r.t. input

    params = list(seq.parameters())
    # Two Linear layers each expose weight (+ bias) => 4 parameter objects expected
    assert len(params) == 4
    for p in params:
        # Each parameter wrapper must expose `.data` and `.grad`
        assert hasattr(p, "data") and hasattr(p, "grad")
        # Name is optional in this test (left None by default)
        assert getattr(p, "name", None) in (None, "weight", "bias")
