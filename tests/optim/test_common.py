import pytest
import nova
import numpy as np
from nova.nn import Parameter
from nova.optim import SGD, Adam


class TestOptimizerCommon:
    """Tests common to all optimizers."""

    def test_param_groups(self):
        """Test that parameter groups work correctly."""
        p1 = Parameter(nova.tensor([1.0]))
        p2 = Parameter(nova.tensor([2.0]))

        optimizer = SGD([{"params": [p1], "lr": 0.1}, {"params": [p2]}], lr=0.01)

        p1.grad = np.array([1.0])
        p2.grad = np.array([1.0])
        optimizer.step()

        # Different learning rates should produce different steps
        step1 = abs(1.0 - p1.data[0])
        step2 = abs(2.0 - p2.data[0])

        assert step1 > step2 * 5  # First group has 10x larger LR

    def test_state_persistence(self):
        """Test that optimizer state persists across steps."""
        p = Parameter(nova.tensor([1.0]))
        optimizer = Adam([p], lr=0.1)

        p.grad = np.array([1.0])
        optimizer.step()

        # Check state was created
        assert p in optimizer.state
        assert optimizer.state[p]["step"] == 1

        p.grad = np.array([1.0])
        optimizer.step()

        # State should update
        assert optimizer.state[p]["step"] == 2

    def test_zero_grad_utility(self):
        """Test zero_grad helper method."""
        p = Parameter(nova.tensor([1.0]))
        optimizer = SGD([p], lr=0.1)

        p.grad = np.array([5.0])
        optimizer.zero_grad()

        assert p.grad is None or np.all(p.grad == 0)
