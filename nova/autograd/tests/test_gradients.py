import nova
import math
import pytest
import numpy as np
import nova.nn as nn
from nova.utils import grad_check_wrt_inputs

nova.manual_seed(8)


def test_gradient_wrt_inputs():

    def operation(input: nova.Tensor) -> nova.Tensor:

        input.flatten()

        res = nova.mean(nova.abs(input - math.e))
        res = nova.pow(res, nova.exp(res))
        return res

    x = nova.tensor([0.3, 0.7, 0.1, 0.3, 10.0], dtype=nova.float32, requires_grad=True)

    analitycal, numerical = grad_check_wrt_inputs(
        operation, x, eps=1e-4, zero_grads=True
    )
    assert nova.allclose(
        analitycal[0], numerical[0], rtol=1e-2, atol=1e-5
    )  # by float32 precision


def test_gradient_wrt_layer_op():

    x = nova.randn(8, 8, dtype=nova.float32, requires_grad=True)

    linear_op = nn.Linear(8, 16)
    batchnorm_op = nn.BatchNorm1d(8)
    layer_norm = nn.LayerNorm(normalized_shape=8, eps=1e-5)

    ops = [linear_op, batchnorm_op, layer_norm]

    for op in ops:
        print("Operation: \n", op)
        analitycal, numerical = grad_check_wrt_inputs(op, x)
        assert nova.allclose(
            analitycal[0], numerical[0], rtol=1e-2, atol=1e-2  # by float32 precision
        ), f"Op '{op.__class__.__name__}' failed"


def test_retain_grad():

    x = nova.randn(8, 8, dtype=nova.float32)
    linear = nn.Linear(8, 16, bias=False)
    bias = nova.ones((16,), dtype=nova.float32, requires_grad=True)

    out = linear(x)

    assert out.grad is None
    assert bias.grad is None

    bias_view = bias.view(1, 16)
    out = out + bias_view
    loss = nova.mean(out)
    out.retain_grad()

    loss.backward()

    assert out.grad is not None
