import numpy as np
import pytest

from src.layers.bn.batch_normalization import BatchNormalization
from src.utils import numeric_grad_elementwise

RNG = np.random.RandomState(0)


def test_batchnorm_forward_basic_stats_and_running_update():
    F = 8
    B = 512
    X = RNG.randn(F, B)

    bn = BatchNormalization(F)
    bn.train()

    out = bn.forward(X)
    assert out.shape == X.shape

    feature_means = np.mean(out, axis=1, keepdims=True)
    feature_vars = np.var(out, axis=1, keepdims=True)

    assert np.allclose(feature_means, 0, atol=1e-1)
    assert np.all(feature_vars <= 3.0)


def test_batchnorm_eval_uses_running_stats():
    F = 6
    B_train = 512
    B_eval = 128
    X_train = RNG.randn(F, B_train)
    X_eval = RNG.randn(F, B_eval)

    bn = BatchNormalization(F)
    bn.train()
    bn.forward(X_train)

    bn.eval()
    out_eval = bn.forward(X_eval)
    mean_out_eval = np.mean(out_eval, axis=1, keepdims=True)
    assert np.all(np.abs(mean_out_eval) < 1.0)
