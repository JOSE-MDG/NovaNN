import pytest
import nova
import numpy as np
from nova.nn import Parameter
from nova.optim import SGD

nova.manual_seed(42)


class TestSGD:
    def test_basic_step(self):
        """Test that SGD performs basic gradient descent."""
        p = Parameter(nova.tensor([10.0]))
        optimizer = SGD([p], lr=0.1)

        # Gradient pointing toward zero
        p.grad = np.array([1.0], dtype=nova.float32)
        optimizer.step()

        # Should move in negative gradient direction
        assert p.data[0] < 10.0
        assert np.isclose(p.data[0], 9.9)

    def test_momentum_accumulation(self):
        """Test that momentum accumulates velocity."""
        p = Parameter(nova.tensor([5.0]))
        optimizer = SGD([p], lr=0.1, momentum=0.9)

        # Apply same gradient multiple times
        for _ in range(3):
            p.grad = np.array([1.0])
            optimizer.step()

        # With momentum, should move further than without
        # First step: v=1.0, x -= 0.1*1.0 = 4.9
        # Second step: v=0.9*1.0+1.0=1.9, x -= 0.1*1.9 = 4.71
        # Third step: v=0.9*1.9+1.0=2.71, x -= 0.1*2.71 ≈ 4.439
        assert p.data[0] < 4.5

    def test_weight_decay(self):
        """Test L2 weight decay."""
        p = Parameter(nova.tensor([1.0]))
        optimizer = SGD([p], lr=0.1, weight_decay=0.1)

        p.grad = np.array([0.0])  # Zero gradient
        initial = p.data[0]
        optimizer.step()

        # With weight decay, parameter should decrease
        assert p.data[0] < initial

    def test_convergence_on_quadratic(self):
        """Test convergence on simple quadratic: f(x) = x^2."""
        p = Parameter(nova.tensor([3.0]))
        optimizer = SGD([p], lr=0.1)

        for _ in range(50):
            # Gradient of x^2 is 2x
            p.grad = 2.0 * p.data
            optimizer.step()

        # Should converge close to zero
        assert abs(p.data[0]) < 0.1

    def test_multiple_parameters(self):
        """Test optimization of multiple parameters."""
        p1 = Parameter(nova.tensor([1.0]))
        p2 = Parameter(nova.tensor([2.0]))
        optimizer = SGD([p1, p2], lr=0.1)

        p1.grad = np.array([1.0])
        p2.grad = np.array([2.0])
        optimizer.step()

        assert np.isclose(p1.data[0], 0.9)
        assert np.isclose(p2.data[0], 1.8)

    def test_zero_grad_handling(self):
        """Test that None gradients are skipped."""
        p = Parameter(nova.tensor([1.0]))
        optimizer = SGD([p], lr=0.1)

        p.grad = None
        initial = p.data.copy()
        optimizer.step()

        # Parameter should not change
        assert np.array_equal(p.data, initial)
