import numpy as np
import pytest

from novann.layers import Dropout

RNG = np.random.RandomState(0)


def test_dropout_eval_pass_through_and_train_masks_scaling():
    # Use a non-zero input to simplify mask inspection
    X = np.ones((4, 5), dtype=np.float32) * 2.0
    p = 0.5
    d = Dropout(p)

    # eval mode: forward returns input unchanged (no dropout applied)
    d.eval()
    out_eval = d.forward(X)
    assert np.allclose(out_eval, X)

    # training mode: mask applied and surviving units scaled by 1/(1-p)
    d.train()
    np.random.seed(123)  # deterministically sample mask for this test
    out_train = d.forward(X.copy())

    # recover mask from outputs (non-zero => kept)
    mask = out_train != 0.0
    # where kept, outputs equal input scaled by 1/(1-p)
    assert np.allclose(out_train[mask], X[mask] / (1 - p))
    # where dropped, outputs are exactly zero
    assert np.all(out_train[~mask] == 0.0)

    # backward in training: gradients are masked and scaled identically
    upstream = np.ones_like(out_train)
    db = d.backward(upstream)
    assert np.allclose(db[mask], upstream[mask] / (1 - p))
    assert np.all(db[~mask] == 0.0)
