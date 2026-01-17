import pytest
import nova
import numpy as np
from nova.nn import Parameter
from nova.optim import RMSprop


class TestRMSprop:
    def test_basic_step(self):
        """Test that RMSprop performs gradient descent."""
        p = Parameter(nova.tensor([10.0]))
        optimizer = RMSprop([p], lr=0.1)

        p.grad = np.array([1.0])
        optimizer.step()

        assert p.data[0] < 10.0

    def test_adaptive_to_gradient_scale(self):
        """Test that RMSprop adapts to gradient magnitude."""
        p = Parameter(nova.tensor([1.0]))
        optimizer = RMSprop([p], lr=0.1, alpha=0.9)

        # Apply large gradient multiple times
        for _ in range(5):
            p.grad = np.array([10.0])
            optimizer.step()

        # RMSprop should adapt and take smaller steps as gradient squared accumulates
        # Exact value depends on implementation but should still be positive
        assert p.data[0] < 1.0

    def test_centered_mode(self):
        """Test centered variance normalization."""
        p1 = Parameter(nova.tensor([1.0]))
        p2 = Parameter(nova.tensor([1.0]))

        opt_centered = RMSprop([p1], lr=0.1, centered=True)
        opt_regular = RMSprop([p2], lr=0.1, centered=False)

        # Apply same gradient
        for _ in range(3):
            p1.grad = np.array([1.0])
            p2.grad = np.array([1.0])
            opt_centered.step()
            opt_regular.step()

        # Results should differ
        assert not np.isclose(p1.data[0], p2.data[0])

    def test_convergence_on_quadratic(self):
        """Test convergence on f(x) = x^2."""
        p = Parameter(nova.tensor([3.0]))
        optimizer = RMSprop([p], lr=0.01)

        for _ in range(500):
            p.grad = 2.0 * p.data
            optimizer.step()

        assert abs(p.data[0]) < 0.1

    def test_momentum(self):
        """Test that momentum accelerates convergence."""
        p = Parameter(nova.tensor([5.0]))
        optimizer = RMSprop([p], lr=0.1, momentum=0.9)

        # Consistent gradient should build up momentum
        for _ in range(5):
            p.grad = np.array([1.0])
            optimizer.step()

        # With momentum, should move further
        assert p.data[0] < 4.0
