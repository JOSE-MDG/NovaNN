import nova
import pytest
import numpy as np
from nova.autograd.grad import grad


def test_size():
    x = nova.randn(2, 3, 4)
    assert x.size() == (2, 3, 4)
    assert x.size(0) == 2
    assert x.size(1) == 3
    assert x.size(-1) == 4


def test_dim():
    x = nova.randn(2, 3, 4)
    assert x.dim() == 3


def test_numel():
    x = nova.randn(2, 3, 4)
    assert x.numel() == 24


def test_clone():
    x = nova.tensor([1, 2, 3], requires_grad=True)
    y = x.clone()
    assert not y.is_leaf
    assert y.requires_grad


def test_grad_function():
    x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x**2).sum()
    grads = grad(y, x)
    expected = np.array([2.0, 4.0, 6.0])
    assert nova.allclose(grads, expected)


def test_grad_mode():
    x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
    with nova.no_grad():
        y = nova.sum(2 * x)

    assert y.requires_grad == False
