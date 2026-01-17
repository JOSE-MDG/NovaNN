import pytest
import nova
import numpy as np
from nova.nn import Parameter
from nova.optim import AdamW, Adam


class TestAdamW:
    def test_basic_step(self):
        """Test that AdamW performs gradient descent."""
        p = Parameter(nova.tensor([10.0]))
        optimizer = AdamW([p], lr=0.1)

        p.grad = np.array([1.0])
        optimizer.step()

        assert p.data[0] < 10.0

    def test_decoupled_weight_decay(self):
        """Test that AdamW uses decoupled weight decay."""
        p = Parameter(nova.tensor([1.0]))
        optimizer = AdamW([p], lr=0.1, weight_decay=0.1)

        # Zero gradient - only weight decay should apply
        p.grad = np.array([0.0])
        optimizer.step()

        # Decoupled weight decay: w = w - lr * wd * w
        # w = 1.0 - 0.1 * 0.1 * 1.0 = 0.99
        assert np.isclose(p.data[0], 0.99, atol=1e-6)

    def test_adamw_vs_adam_with_weight_decay(self):
        """Test that AdamW and Adam differ when using weight decay."""
        p1 = Parameter(nova.tensor([1.0]))
        p2 = Parameter(nova.tensor([1.0]))

        adam = Adam([p1], lr=0.1, weight_decay=0.1)
        adamw = AdamW([p2], lr=0.1, weight_decay=0.1)

        # Apply same gradient
        p1.grad = np.array([1.0])
        p2.grad = np.array([1.0])

        adam.step()
        adamw.step()

        # Should produce different results due to decoupled vs coupled WD
        assert not np.isclose(p1.data[0], p2.data[0])

    def test_convergence_on_quadratic(self):
        """Test convergence on f(x) = x^2."""
        p = Parameter(nova.tensor([3.0]))
        optimizer = AdamW([p], lr=0.1)

        for _ in range(200):
            p.grad = 2.0 * p.data
            optimizer.step()

        assert abs(p.data[0]) < 0.01

    def test_bias_correction(self):
        """Test bias correction in first steps."""
        p = Parameter(nova.tensor([1.0]))
        optimizer = AdamW([p], lr=1.0)

        p.grad = np.array([1.0])
        optimizer.step()

        # Should take significant first step with bias correction
        assert p.data[0] < 0.7
