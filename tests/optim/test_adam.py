import pytest
import nova
import numpy as np
from nova.nn import Parameter
from nova.optim import Adam

nova.manual_seed(42)


class TestAdam:
    def test_basic_step(self):
        """Test that Adam performs gradient descent."""
        p = Parameter(nova.tensor([10.0]))
        optimizer = Adam([p], lr=0.1)

        p.grad = np.array([1.0])
        optimizer.step()

        # Should move in negative gradient direction
        assert p.data[0] < 10.0

    def test_bias_correction(self):
        """Test that bias correction is applied in early steps."""
        p = Parameter(nova.tensor([1.0]))
        optimizer = Adam([p], lr=1.0, betas=(0.9, 0.999))

        p.grad = np.array([1.0])
        optimizer.step()

        # With bias correction, first step should be larger
        # m_hat = 0.1 / (1 - 0.9) = 1.0
        # v_hat = 0.01 / (1 - 0.999) ≈ 10.0
        # update ≈ 1.0 / sqrt(10.0) ≈ 0.316
        assert p.data[0] < 0.7  # Should take significant step

    def test_adaptive_learning_rate(self):
        """Test that Adam adapts to gradient magnitude."""
        p1 = Parameter(nova.tensor([1.0]))
        p2 = Parameter(nova.tensor([1.0]))

        opt1 = Adam([p1], lr=0.01)
        opt2 = Adam([p2], lr=0.01)

        # Small gradient
        p1.grad = np.array([0.1])
        opt1.step()

        # Large gradient
        p2.grad = np.array([10.0])
        opt2.step()

        # Step sizes should differ due to adaptive scaling
        step1 = abs(1.0 - p1.data[0])
        step2 = abs(1.0 - p2.data[0])

        # Step sizes should be closer than gradient ratio (100x)
        assert step2 / step1 < 50  # Adaptive scaling reduces the difference

    def test_convergence_on_quadratic(self):
        """Test convergence on f(x) = x^2."""
        p = Parameter(nova.tensor([3.0]))
        optimizer = Adam([p], lr=0.1)

        for _ in range(200):
            p.grad = 2.0 * p.data
            optimizer.step()

        assert abs(p.data[0]) < 0.01

    def test_weight_decay_coupled(self):
        """Test that Adam uses coupled weight decay."""
        p = Parameter(nova.tensor([1.0]))
        optimizer = Adam([p], lr=0.1, weight_decay=0.1)

        p.grad = np.array([0.0])
        optimizer.step()

        # Weight decay should affect the parameter
        assert p.data[0] < 1.0
