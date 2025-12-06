import numpy as np
import pytest

from novann.layers import BatchNorm1d

RNG = np.random.RandomState(0)


def test_batchnorm_forward_basic_stats_and_running_update():
    F = 8
    B = 512
    X = RNG.randn(B, F)

    bn = BatchNorm1d(F)
    bn.train()

    out = bn(X)
    # shape preserved (batch, features)
    assert out.shape == X.shape

    feature_means = np.mean(out, axis=0, keepdims=True)
    feature_vars = np.var(out, axis=0, keepdims=True)

    # After normalization the per-feature mean should be near zero (loose tolerance)
    assert np.allclose(feature_means, 0, atol=1e-1)
    # Variance should be reasonably bounded after normalization
    assert np.all(feature_vars <= 3.0)


def test_batchnorm_eval_uses_running_stats():
    F = 6
    B_train = 512
    B_eval = 128
    X_train = RNG.randn(B_train, F)
    X_eval = RNG.randn(B_eval, F)

    bn = BatchNorm1d(F)
    bn.train()
    # update running statistics using a training batch
    bn.forward(X_train)

    bn.eval()
    out_eval = bn.forward(X_eval)
    mean_out_eval = np.mean(out_eval, axis=0, keepdims=True)
    # In eval mode outputs should be roughly centered (using running stats)
    assert np.all(np.abs(mean_out_eval) < 1.0)
