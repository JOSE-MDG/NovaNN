import nova
import pytest
import numpy as np
import nova.nn as nn
from nova.autograd.utils.gradients import grad_check_wrt_inputs


def test_grad_wrt_input():
    X = nova.randn(16, 8, dtype=nova.float32)
    model = nn.Sequential(nn.Linear(8, 12), nn.ReLU(), nn.Linear(12, 2), nn.Softmax())

    print("Model: \n", model)
    forward = lambda x=X, model=model: model(x)

    analytcal, numerical = grad_check_wrt_inputs(forward, x=X, model=model)

    assert nova.allclose(analytcal[0], numerical[0], rtol=1e-3, atol=1e-5)
