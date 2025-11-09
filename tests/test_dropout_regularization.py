import numpy as np
import pytest

from src.layers.regularization.dropout import Dropout

RNG = np.random.RandomState(0)


def test_dropout_eval_pass_through_and_train_masks_scaling():
    X = np.ones((4, 5), dtype=np.float32) * 2.0  # avoid zeros so mask recovery simple
    p = 0.5
    d = Dropout(p)

    # eval mode: forward returns input unchanged
    d.eval()
    out_eval = d.forward(X)
    assert np.allclose(out_eval, X)

    # training mode: mask applied and scaling by 1/(1-p)
    d.train()
    np.random.seed(123)
    out_train = d.forward(X.copy())

    # recovered mask: True where output non-zero
    mask = out_train != 0.0
    # where mask True, out should equal x / (1 - p)
    assert np.allclose(out_train[mask], X[mask] / (1 - p))
    # where mask False, out should be zero
    assert np.all(out_train[~mask] == 0.0)

    # backward in training: grad multiplied by same mask and scaled
    upstream = np.ones_like(out_train)
    db = d.backward(upstream)
    assert np.allclose(db[mask], upstream[mask] / (1 - p))
    assert np.all(db[~mask] == 0.0)  # '~' boolean operator
