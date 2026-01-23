import nova
import pytest
import numpy as np
from nova.autograd.engine.engine import _build_topo, _backward


def test_topo_order():

    x1 = nova.tensor([1.0, 0.2, 0.3, 4.1], requires_grad=True)
    x2 = nova.tensor([0.1, 0.4, 2.9, 0.34], requires_grad=True)

    x3 = x1 + x2
    x4 = nova.mean(x3, dim=0)
    x5 = x4**x1

    correct_order = [x1, x2, x3, x4, x5]
    constructed_topo = _build_topo(x5)

    for ord, topo in zip(correct_order, constructed_topo):
        assert ord is topo, "The topological sort is incorrect"


def test_parents_of_tensors():

    x1 = nova.tensor([1.0, 0.5, 0.3, 8.1], requires_grad=True)
    x2 = nova.tensor([0.15, 3.4, 2.4, 5.34], requires_grad=True)

    x3 = x1.pow(x2)
    x4 = x3 + x2
    x5 = x4.sqrt()
    x6 = x5.sum()

    x3_parents = [x1, x2]
    x4_parents = [x3, x2]
    x5_parents = [x4]
    x6_parents = [x5]

    tensors = [x3, x4, x5, x6]
    parents = [x3_parents, x4_parents, x5_parents, x6_parents]

    for t, p in zip(tensors, parents):
        assert (
            t._inputs == p
        ), "The tensioners are incorrectly storing their parents tensors"


def test_backward_pass_exceptions():

    x1 = nova.tensor([1.5, 0.6, 3.3, 0.1])
    x2 = nova.tensor([0.54, 1.3, 4.4, 5.0])

    with pytest.raises(RuntimeError):
        x1.backward([1, 1, 1, 1])

    with pytest.raises(RuntimeError):
        x2.backward([1, 1, 1, 1])

    x1 = nova.tensor([1.5, 0.6, 3.3, 0.1], requires_grad=True)
    x2 = nova.tensor([0.54, 1.3, 4.4, 5.0], requires_grad=True)

    with pytest.raises(RuntimeError):
        x1.backward()

    with pytest.raises(RuntimeError):
        x2.backward()

    with pytest.raises(RuntimeError):
        x1 *= 10

    with pytest.raises(RuntimeError):
        x2 *= 20

    with pytest.raises(Exception):
        x1.backward(["hello"])


def test_backward_pass():

    x1 = nova.tensor([1.5, -0.7, 3.3, 2.2], requires_grad=True)
    x2 = nova.tensor([9.54, -6.32, -4.4, 7.0], requires_grad=True)

    assert x1.grad is None
    assert x2.grad is None

    x3 = nova.exp(x2)
    x4 = x3 - x1.minimum(1e-20)
    x5 = nova.sum(x4 ** nova.sin(x1))

    gradient = np.array([1], dtype=nova.float32)
    _backward(x5, gradient)

    assert x1.grad is not None
    assert x2.grad is not None


def test_retain_graph_simple():
    x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x**2).sum()

    y.backward(retain_graph=True)
    grad_first = x.grad.copy()

    x.zero_grad()
    y.backward(retain_graph=True)
    grad_second = x.grad.copy()

    assert nova.allclose(grad_first, grad_second)
    assert nova.allclose(x.grad, np.array([2.0, 4.0, 6.0]))


def test_gradient_accumulation():
    x = nova.tensor([1.0, 2.0], requires_grad=True)

    y1 = (x**2).sum()
    y2 = (x**3).sum()

    y1.backward(retain_graph=True)
    grad_after_y1 = x.grad.copy()

    y2.backward()
    grad_after_both = x.grad.copy()

    expected_y1 = np.array([2.0, 4.0])  # d/dx(x²) = 2x
    expected_y2 = np.array([3.0, 12.0])  # d/dx(x³) = 3x²
    expected_total = expected_y1 + expected_y2

    assert nova.allclose(grad_after_y1, expected_y1)
    assert nova.allclose(grad_after_both, expected_total)


def test_shared_computation_graph():
    x1 = nova.tensor([1.0, 2.0], requires_grad=True)
    x2 = nova.tensor([3.0, 4.0], requires_grad=True)

    shared = x1 + x2
    y1 = shared.sum()
    y2 = (shared**2).sum()

    y1.backward(retain_graph=True)
    y2.backward()

    # y1: d/dx1 = 1, d/dx2 = 1
    # y2: d/dx1 = 2*shared = 2*(x1+x2), d/dx2 = 2*shared
    expected_x1_y1 = np.array([1.0, 1.0])
    expected_x1_y2 = 2 * (x1.data + x2.data)
    expected_x1_total = expected_x1_y1 + expected_x1_y2

    assert nova.allclose(x1.grad, expected_x1_total)
    assert nova.allclose(x2.grad, expected_x1_total)


def test_no_retain_graph_fails():
    x = nova.tensor([1.0, 2.0], requires_grad=True)
    y = (x**2).sum()

    y.backward()

    x.zero_grad()
    try:
        y.backward()
        assert x.grad is None or nova.allclose(x.grad, np.zeros_like(x.data))
    except (RuntimeError, AttributeError):
        pass


def test_retain_graph_with_zero_grad():
    x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x**2).sum()

    y.backward(retain_graph=True)
    first_grad = x.grad.copy()

    x.zero_grad()
    assert x.grad is None or nova.allclose(x.grad, np.zeros_like(x.data))

    y.backward()
    second_grad = x.grad.copy()

    assert nova.allclose(first_grad, second_grad)


def test_mean_plus_sum_accumulation():
    x = nova.tensor([1.0, 2.0], requires_grad=True)

    z = x**2

    y_mean = z.mean()
    y_sum = z.sum()

    y_mean.backward(retain_graph=True)
    grad_mean = x.grad.copy()

    y_sum.backward()
    grad_total = x.grad.copy()

    assert not nova.allclose(grad_mean, grad_total)

    # mean: d/dx(mean(x²)) = 2x/n
    # sum: d/dx(sum(x²)) = 2x
    n = len(x.data)
    expected_mean = 2 * x.data / n
    expected_sum = 2 * x.data
    expected_total = expected_mean + expected_sum

    assert nova.allclose(grad_total, expected_total, rtol=1e-4)
