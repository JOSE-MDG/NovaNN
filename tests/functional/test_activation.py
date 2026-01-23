import nova
import pytest
import nova.nn as nn
from nova.utils import grad_check_wrt_inputs

# activations: relu, leaky_rely, gelu, prelu, tanh, sigmoid, softmax, log_softmax

nova.manual_seed(8)


def test_relu_forward():

    x = nova.randn(8, 8, dtype=nova.float32)

    act = nn.ReLU()
    out = act(x)

    assert nova.all(out >= 0)
    assert out.shape == x.shape


def test_relu_backward():

    x = nova.randn(8, 8, dtype=nova.float32, requires_grad=True)

    act = nn.ReLU()

    analitycal, numerical = grad_check_wrt_inputs(act, x, eps=1e-5)

    assert nova.allclose(analitycal[0], numerical[0], rtol=1e-3, atol=5e-3)


# LeakyReLU


def test_leaky_relu_forward():
    x = nova.randn(8, 8, dtype=nova.float32)

    # Use a distinct negative slope to verify logic
    act = nn.LeakyReLU(negative_slope=0.1)
    out = act(x)

    assert out.shape == x.shape
    # Ensure negative values are scaled
    assert nova.all(out[x < 0] < 0)


def test_leaky_relu_backward():
    x = nova.randn(8, 8, dtype=nova.float32, requires_grad=True)
    act = nn.LeakyReLU(negative_slope=0.1)

    # Gradient consistency check
    analytical, numerical = grad_check_wrt_inputs(act, x, eps=1e-5)
    assert nova.allclose(analytical[0], numerical[0], rtol=1e-3, atol=5e-3)


# GELU


def test_gelu_forward():
    x = nova.randn(8, 8, dtype=nova.float32)

    act = nn.GELU()
    out = act(x)

    assert out.shape == x.shape


def test_gelu_backward():
    x = nova.randn(8, 8, dtype=nova.float32, requires_grad=True)
    act = nn.GELU()

    # Gradient consistency check
    analytical, numerical = grad_check_wrt_inputs(act, x, eps=1e-4)
    assert nova.allclose(analytical[0], numerical[0], rtol=1e-2, atol=5e-2)


# PReLU


def test_prelu_forward():
    x = nova.randn(8, 8, dtype=nova.float32)

    # PReLU has learnable parameters
    act = nn.PReLU()
    out = act(x)

    assert out.shape == x.shape


def test_prelu_backward():
    x = nova.randn(8, 8, dtype=nova.float32, requires_grad=True)
    act = nn.PReLU()

    # Gradient consistency check w.r.t input x
    analytical, numerical = grad_check_wrt_inputs(act, x, eps=1e-4)
    assert nova.allclose(analytical[0], numerical[0], rtol=1e-2, atol=5e-2)


# Tanh


def test_tanh_forward():
    x = nova.randn(8, 8, dtype=nova.float32)

    act = nn.Tanh()
    out = act(x)

    assert out.shape == x.shape
    # Output range must be (-1, 1)
    assert nova.all(out >= -1) and nova.all(out <= 1)


def test_tanh_backward():
    x = nova.randn(8, 8, dtype=nova.float32, requires_grad=True)
    act = nn.Tanh()

    # Gradient consistency check
    analytical, numerical = grad_check_wrt_inputs(act, x, eps=1e-4)
    assert nova.allclose(analytical[0], numerical[0], rtol=1e-3, atol=5e-3)


# Sigmoid


def test_sigmoid_forward():
    x = nova.randn(8, 8, dtype=nova.float32)

    act = nn.Sigmoid()
    out = act(x)

    assert out.shape == x.shape
    # Output range must be (0, 1)
    assert nova.all(out >= 0) and nova.all(out <= 1)


def test_sigmoid_backward():
    x = nova.randn(8, 8, dtype=nova.float32, requires_grad=True)
    act = nn.Sigmoid()

    # Gradient consistency check
    analytical, numerical = grad_check_wrt_inputs(act, x, eps=1e-5)
    assert nova.allclose(analytical[0], numerical[0], rtol=5e-3, atol=5e-3)


# Softmax


def test_softmax_forward():
    x = nova.randn(4, 10, dtype=nova.float32)

    act = nn.Softmax(dim=1)
    out = act(x)

    assert out.shape == x.shape
    # Check probabilities sum to 1 along dim
    sums = out.sum(dim=1)
    assert nova.allclose(sums, nova.ones_like(sums), atol=1e-4)
    assert nova.all(out >= 0) and nova.all(out <= 1)


def test_softmax_backward():
    x = nova.randn(4, 10, dtype=nova.float32, requires_grad=True)
    act = nn.Softmax(dim=1)

    # Gradient consistency check
    analytical, numerical = grad_check_wrt_inputs(act, x, eps=1e-4)
    assert nova.allclose(analytical[0], numerical[0], rtol=1e-3, atol=5e-2)


# LogSoftmax


def test_log_softmax_forward():
    x = nova.randn(4, 10, dtype=nova.float32)

    act = nn.LogSoftmax(dim=1)
    out = act(x)

    assert out.shape == x.shape
    # exp(out) should sum to 1
    probs = nova.exp(out)
    sums = probs.sum(dim=1)
    assert nova.allclose(sums, nova.ones_like(sums), atol=1e-4)


def test_log_softmax_backward():
    x = nova.randn(4, 10, dtype=nova.float32, requires_grad=True)
    act = nn.LogSoftmax(dim=1)

    # Gradient consistency check
    analytical, numerical = grad_check_wrt_inputs(act, x, eps=1e-4)
    assert nova.allclose(analytical[0], numerical[0], rtol=1e-3, atol=5e-2)
