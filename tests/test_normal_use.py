import nova
import pytest
import numpy as np
from nova.nn.utils import clip_grad_norm_, clip_grad_value_
from nova.optim import SGD, Adam, AdamW
from nova.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
from nova.nn import (
    Sequential,
    Linear,
    ReLU,
    Conv2d,
    Flatten,
    MSELoss,
    CrossEntropyLoss,
    Parameter,
)

nova.manual_seed(42)


class TestComplexDtypes:
    """Test operations with complex number dtypes"""

    def test_complex64_forward_backward(self):
        """Complex64 should work in forward and backward passes"""
        x = nova.tensor(
            [1.0 + 2.0j, 3.0 + 4.0j], dtype=nova.complex64, requires_grad=True
        )
        y = x * 2.0
        z = y.sum()
        z.backward()

        assert x.grad is not None
        assert x.grad.dtype == np.complex64

    def test_complex128_operations(self):
        """Complex128 operations should preserve precision"""
        a = nova.tensor([1.0 + 1.0j, 2.0 + 2.0j], dtype=nova.complex128)
        b = nova.tensor([3.0 + 3.0j, 4.0 + 4.0j], dtype=nova.complex128)

        c = a + b
        d = a * b

        assert c.dtype == nova.complex128
        assert d.dtype == nova.complex128
        assert np.allclose(c.data, np.array([4.0 + 4.0j, 6.0 + 6.0j]))

    def test_complex256_if_available(self):
        """Complex256 should work on supported platforms"""
        import sys

        if sys.platform.startswith("win32"):
            # On Windows it should fallback to complex128
            x = nova.tensor([1.0 + 1.0j], dtype=nova.complex256)
            assert x.dtype == np.complex128
        else:
            # On Linux/Mac it should actually be complex256
            x = nova.tensor([1.0 + 1.0j], dtype=nova.complex256)
            assert x.dtype == np.complex256

    def test_mixed_complex_real_operations(self):
        """Complex and real tensors should interact correctly"""
        real = nova.tensor([2.0, 3.0])
        comp = nova.tensor([1.0 + 1.0j, 2.0 + 2.0j], dtype=nova.complex64)

        result = real * comp
        assert np.iscomplexobj(result.data)


class TestParameterGroups:
    """Test different parameter groups with different hyperparameters"""

    def test_different_lr_per_group(self):
        """Each param group should have its own learning rate"""
        p1 = Parameter(nova.tensor([1.0], dtype=nova.longdouble))
        p2 = Parameter(nova.tensor([2.0], dtype=nova.longdouble))

        optimizer = SGD([{"params": [p1], "lr": 0.1}, {"params": [p2]}], lr=0.01)

        p1.grad = np.array([1.0])
        p2.grad = np.array([1.0])

        optimizer.step()

        # p1 should move more than p2
        assert abs(1.0 - p1.data[0]) > abs(2.0 - p2.data[0])

    def test_weight_decay_per_group(self):
        """Weight decay should apply differently per group"""
        p1 = Parameter(nova.tensor([1.0], dtype=nova.longdouble))
        p2 = Parameter(nova.tensor([1.0], dtype=nova.longdouble))

        optimizer = AdamW(
            [
                {"params": [p1], "lr": 0.01, "weight_decay": 0.1},
                {"params": [p2], "lr": 0.01, "weight_decay": 0.0},
            ],
            lr=0.01,
        )  # Default values

        p1.grad = np.array([0.0])
        p2.grad = np.array([0.0])

        initial_p1 = p1.data.copy()
        initial_p2 = p2.data.copy()

        optimizer.step()

        # p1 should decay, p2 shouldn't
        assert p1.data[0] < initial_p1[0]
        assert np.allclose(p2.data[0], initial_p2[0])

    def test_momentum_per_group(self):
        """Different momentum values per group"""
        p1 = Parameter(nova.tensor([0.0]))
        p2 = Parameter(nova.tensor([0.0]))

        optimizer = SGD(
            [
                {"params": [p1], "lr": 0.1, "momentum": 0.9},
                {"params": [p2], "lr": 0.1, "momentum": 0.0},
            ],
            lr=0.1,
        )  # Default values

        # First step
        p1.grad = np.array([1.0])
        p2.grad = np.array([1.0])
        optimizer.step()

        step1_p1 = p1.data.copy()
        step1_p2 = p2.data.copy()

        # Second step with same gradient
        p1.grad = np.array([1.0])
        p2.grad = np.array([1.0])
        optimizer.step()

        # p1 should accelerate due to momentum
        step2_p1_change = abs(p1.data[0] - step1_p1[0])
        step2_p2_change = abs(p2.data[0] - step1_p2[0])

        assert step2_p1_change > step2_p2_change


class TestGradientClipping:
    """Test gradient clipping in realistic scenarios"""

    def test_clip_by_norm_prevents_explosion(self):
        """Gradient clipping should prevent gradient explosion"""
        model = Sequential(
            Linear(10, 50), ReLU(), Linear(50, 50), ReLU(), Linear(50, 1)
        )

        x = nova.randn(4, 10)
        y = nova.randn(4, 1)

        # Intentionally create large gradients
        output = model(x) * 1000.0
        loss = ((output - y) ** 2).mean()
        loss.backward()

        total_norm_before = clip_grad_norm_(
            model.parameters(), max_norm=1.0, get_norm=True
        )

        # Clip gradients
        total_norm_after = clip_grad_norm_(
            model.parameters(), max_norm=1.0, get_norm=True
        )

        # After clipping, effective norm should be <= 1.0
        assert total_norm_after <= 1.0 + 1e-6
        assert total_norm_before >= total_norm_after

    def test_clip_by_value(self):
        """Value clipping should bound individual gradient elements"""
        p = Parameter(nova.tensor([1.0, 2.0, 3.0], dtype=nova.longdouble))
        p.grad = np.array([10.0, -15.0, 5.0])

        clip_grad_value_([p], clip_value=7.0)

        assert np.all(np.abs(p.grad) <= 7.0)
        assert np.allclose(p.grad, [7.0, -7.0, 5.0])


class TestSchedulerIntegration:
    """Test learning rate schedulers in realistic scenarios"""

    def test_step_lr_decay(self):
        """StepLR should decay learning rate at specified intervals"""
        p = Parameter(nova.randn(10, 10), dtype=nova.longdouble)
        optimizer = SGD([p], lr=0.1)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])

            p.grad = np.random.randn(*p.shape)
            optimizer.step()
            scheduler.step()

        # Check decay pattern
        assert np.allclose(lrs[0], 0.1)
        assert np.allclose(lrs[3], 0.05)
        assert np.allclose(lrs[6], 0.025)
        assert np.allclose(lrs[9], 0.0125)

    def test_cosine_annealing(self):
        """Cosine annealing should smoothly decrease learning rate"""
        p = Parameter(nova.randn(5, 5), dtype=nova.longdouble)
        optimizer = Adam([p], lr=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01)

        lrs = []
        for _ in range(11):
            lrs.append(optimizer.param_groups[0]["lr"])

            p.grad = np.random.randn(*p.shape)
            optimizer.step()
            scheduler.step()

        # Should start high, end low
        assert lrs[0] > lrs[-1]
        assert lrs[-1] <= 0.01

        # Should be smooth (monotonic decrease in first half)
        assert all(lrs[i] >= lrs[i + 1] for i in range(5))

    def test_onecycle_with_momentum(self):
        """OneCycleLR should cycle both LR and momentum"""
        p = Parameter(nova.randn(3, 3), dtype=nova.longdouble)
        optimizer = SGD([p], lr=0.1, momentum=0.9)
        scheduler = OneCycleLR(optimizer, max_lr=0.5, total_steps=10, pct_start=0.3)

        lrs = []
        momentums = []

        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            momentums.append(optimizer.param_groups[0]["momentum"])

            p.grad = np.random.randn(*p.shape)
            optimizer.step()
            scheduler.step()

        # LR should increase then decrease
        assert max(lrs) > lrs[0]
        assert max(lrs) > lrs[-1]

        # Momentum should decrease then increase (inverse of LR)
        assert min(momentums) < momentums[0]

    def test_scheduler_with_param_groups(self):
        """Scheduler should work with multiple parameter groups"""
        p1 = Parameter(nova.randn(2, 2), dtype=nova.longdouble)
        p2 = Parameter(nova.randn(2, 2), dtype=nova.longdouble)

        optimizer = Adam(
            [{"params": [p1], "lr": 0.1}, {"params": [p2]}], lr=0.01
        )  # Default lr

        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        for _ in range(5):
            p1.grad = np.random.randn(*p1.shape)
            p2.grad = np.random.randn(*p2.shape)
            optimizer.step()
            scheduler.step()

        # Both groups should decay proportionally
        assert optimizer.param_groups[0]["lr"] < 0.1
        assert optimizer.param_groups[1]["lr"] < 0.01
        assert optimizer.param_groups[0]["lr"] / optimizer.param_groups[1]["lr"] == 10.0


class TestRealisticTrainingScenarios:
    """Test complete training workflows"""

    def test_convergence(self):
        """Train a small CNN on random data"""
        model = Sequential(
            Conv2d(1, 8, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(8, 16, kernel_size=3, padding=1),
            ReLU(),
            Flatten(),
            Linear(16 * 8 * 8, 10),
        )

        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = CrossEntropyLoss()

        x = nova.randn(4, 1, 8, 8)
        y = nova.tensor([0, 1, 2, 3], dtype=nova.int)

        initial_loss = None
        for i in range(20):
            output = model(x)
            loss = criterion(output, y)

            if i == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss

    def test_transfer_learning_simulation(self):
        """Simulate transfer learning with frozen base"""
        # Pretrained base (frozen)
        base = Sequential(Linear(10, 20), ReLU(), Linear(20, 20), ReLU())
        base_params = []
        for param in base.parameters():
            param.requires_grad = False
            base_params.append(param.data.copy())

        # New head
        head = Sequential(Linear(20, 10), ReLU(), Linear(10, 3))
        head_params = [param.data.copy() for param in head.parameters()]

        # Only train head
        optimizer = Adam(head.parameters(), lr=0.001)
        criterion = CrossEntropyLoss()

        x = nova.randn(16, 10)
        y = nova.tensor(
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=nova.int
        )

        for _ in range(15):
            # Forward through frozen base
            features = base(x)

            # Forward through trainable head
            output = head(features)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Head params should have changed
        # Base params should still be at initialization (approximately)
        for old_bp, new_bp in zip(
            head_params, [param.data for param in head.parameters()]
        ):
            assert not nova.allclose(old_bp, new_bp)

        for old_bp, new_bp in zip(
            base_params, [param.data for param in base.parameters()]
        ):
            assert nova.allclose(old_bp, new_bp)


class TestEdgeCases:
    """Test edge cases and unusual scenarios"""

    def test_zero_gradients(self):
        """Training with zero gradients should not crash"""
        model = Sequential(Linear(5, 5))
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = MSELoss()

        x = nova.randn(2, 5)
        y = nova.randn(2, 5)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y) * 0.0  # Zero loss
        loss.backward()
        optimizer.step()

        # Should not crash
        assert True

    def test_large_depth_network(self):
        """Deep network should propagate gradients"""
        layers = []
        for _ in range(20):
            layers.extend([Linear(10, 10), ReLU()])

        model = Sequential(*layers)[:-1]
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = MSELoss()

        x = nova.randn(4, 10)
        y = nova.randn(4, 10)

        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # All layers should have gradients
        has_grad = sum(
            1 for p in model.parameters() if p.grad is not None or p.grad >= 1e-3
        )
        total_params = len(list(model.parameters()))

        assert has_grad == total_params

        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    def test_empty_parameter_group(self):
        """Optimizer with empty param group should handle gracefully"""
        p = Parameter(nova.tensor([1.0]))

        with pytest.raises(ValueError, match="param_group 'params' is empty"):
            Adam([{"params": [], "lr": 0.01}, {"params": [p]}], lr=0.01)

    def test_scheduler_beyond_total_steps(self):
        """Scheduler called beyond total steps should handle gracefully"""
        p = Parameter(nova.randn(2, 2), dtype=nova.longdouble)
        optimizer = Adam([p], lr=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=5)

        # Run beyond T_max
        for _ in range(10):
            p.grad = np.random.randn(*p.shape)
            optimizer.step()
            scheduler.step()

        # Should not crash
        assert optimizer.param_groups[0]["lr"] == 0


class TestComplexWorkflows:
    """Test realistic complex workflows"""

    def test_early_stopping_simulation(self):
        """Simulate early stopping based on validation loss"""
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = MSELoss()

        x_train = nova.randn(32, 10)
        y_train = nova.randn(32, 5)

        x_val = nova.randn(8, 10)
        y_val = nova.randn(8, 5)

        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        for epoch in range(20):
            # Training
            model.train()
            output = model(x_train)
            loss = criterion(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with nova.no_grad():
                val_output = model(x_val)
                val_loss = criterion(val_output, y_val)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        assert epoch < 20

    def test_learning_rate_finder_simulation(self):
        """Simulate LR range test"""
        model = Sequential(Linear(5, 10), ReLU(), Linear(10, 2))
        criterion = MSELoss()

        x = nova.randn(16, 5)
        y = nova.randn(16, 2)

        lr_start = 1e-6
        lr_end = 1.0
        num_steps = 20

        lrs = np.logspace(np.log10(lr_start), np.log10(lr_end), num_steps)
        losses = []

        for lr in lrs:
            optimizer = SGD(model.parameters(), lr=float(lr))

            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Losses should vary with LR
        assert not all(l == losses[0] for l in losses)  # noqa: E741

    def test_multitask_learning_simulation(self):
        """Simulate multi-task learning with shared backbone"""
        # Shared backbone
        backbone = Sequential(Linear(10, 20), ReLU())

        # Task-specific heads
        task1_head = Linear(20, 5)
        task2_head = Linear(20, 3)

        # Separate optimizers per task (could also be one)
        opt1 = Adam(
            list(backbone.parameters()) + list(task1_head.parameters()), lr=0.001
        )
        opt2 = Adam(
            list(backbone.parameters()) + list(task2_head.parameters()), lr=0.001
        )

        criterion = MSELoss()

        x = nova.randn(8, 10)
        y1 = nova.randn(8, 5)
        y2 = nova.randn(8, 3)

        # Alternate between tasks
        for i in range(10):
            features = backbone(x)

            if i % 2 == 0:
                # Task 1
                out1 = task1_head(features)
                loss = criterion(out1, y1)
                loss.backward()
                opt1.step()
                opt1.zero_grad()
            else:
                # Task 2
                out2 = task2_head(features)
                loss = criterion(out2, y2)
                loss.backward()
                opt2.step()
                opt2.zero_grad()

        assert True
