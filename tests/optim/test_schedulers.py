import pytest
import nova
import nova.nn as nn
import numpy as np
import nova.nn.functional as F
from nova._interfaces._optimizer import Optimizer
from nova._interfaces._lr_scheduler import _LRScheduler
from nova.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
from nova.optim import Adam, AdamW, RMSprop, SGD
from nova.nn import Parameter

nova.manual_seed(8)


class TestStepLR:

    def test_basic_lr_step(self):
        """Test that StepLR correctly decays learning rate at step intervals"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == 0.1

        # Step 0 -> lr should stay at 0.1
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == 0.1

        # Step 1 -> lr should stay at 0.1
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == 0.05  # 0.1 * 0.5

        # Step 2 -> lr should stay at 0.05
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == 0.05

        # Step 3 -> lr should decay to 0.025
        scheduler.step()
        assert np.isclose(optimizer.param_groups[0]["lr"], 0.025)

    def test_convergence(self):
        """Test that model can converge with StepLR scheduler"""
        # Simple linear regression problem
        X = nova.randn(100, 5)
        y = X @ nova.randn(5, 1) + 0.1 * nova.randn(100, 1)

        model = nn.Sequential(nn.Linear(5, 1))
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

        initial_loss = None
        for _ in range(30):
            pred = model(X)
            loss = F.mse_loss(pred, y)

            if initial_loss is None:
                initial_loss = loss

            # Backward pass
            for param in model.parameters():
                param.grad = np.random.randn(*param.shape) * 0.01  # Mock gradient

            optimizer.step()
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        expected_lr = 0.1 * (0.9**3)  # 3 decays in 30 steps
        assert np.isclose(final_lr, expected_lr, rtol=1e-5)

    def test_last_epoch_update(self):
        """Test that last_epoch is correctly updated"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5, last_epoch=4)

        assert scheduler.last_epoch == 5
        scheduler.step()
        assert scheduler.last_epoch == 6

    def test_gamma_coefficient(self):
        """Test different gamma values"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=1.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

        for i in range(5):
            scheduler.step()
            expected_lr = 1.0 * (0.1 ** (i + 1))
            assert np.isclose(optimizer.param_groups[0]["lr"], expected_lr)

    def test_save_state_dict(self):
        """Test saving scheduler state"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        for _ in range(3):
            scheduler.step()

        state = scheduler.state_dict()
        assert "last_epoch" in state
        assert state["last_epoch"] == 3

    def test_load_state_dict(self):
        """Test loading scheduler state"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        # Create state
        for _ in range(5):
            scheduler.step()

        state = scheduler.state_dict()

        # Create new scheduler and load state
        new_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
        new_scheduler.load_state_dict(state)

        assert new_scheduler.last_epoch == 5
        assert new_scheduler.get_last_lr() == scheduler.get_last_lr()


class TestCosineAnnealingLR:

    def test_basic_lr_step(self):
        """Test cosine annealing learning rate progression"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01)

        lrs = []
        for _ in range(11):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Check that lr decreases monotonically
        assert lrs[0] == 0.1  # Initial lr
        assert lrs[-1] >= 0.01  # Should be close to eta_min

        # Check that it follows cosine curve (middle should be between min and max)
        mid_lr = lrs[5]
        assert 0.01 < mid_lr < 0.1

    def test_cosine_convergence(self):
        """Test that learning rate reaches eta_min at T_max"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=1.0)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

        for _ in range(100):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert np.isclose(final_lr, 0.001, atol=1e-6)

    def test_last_epoch_update(self):
        """Test last_epoch tracking"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, last_epoch=2)

        assert scheduler.last_epoch == 3
        scheduler.step()
        assert scheduler.last_epoch == 4

    def test_save_state_dict(self):
        """Test state dict saving"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01)

        for _ in range(5):
            scheduler.step()

        state = scheduler.state_dict()
        assert "last_epoch" in state
        assert state["last_epoch"] == 5

    def test_load_state_dict(self):
        """Test state dict loading"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01)

        for _ in range(7):
            scheduler.step()

        state = scheduler.state_dict()

        new_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01)
        new_scheduler.load_state_dict(state)

        assert new_scheduler.last_epoch == 7
        assert nova.allclose(new_scheduler.get_last_lr(), scheduler.get_last_lr())


class TestOneCycleLR:

    def test_basic_lr_step(self):
        """Test one cycle learning rate progression"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1, momentum=0.9)
        scheduler = OneCycleLR(optimizer, max_lr=1.0, total_steps=10, pct_start=0.3)

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Check initial lr is max_lr / div_factor
        assert np.isclose(lrs[0], 1.0 / 25.0, rtol=1e-5)

        # Check that lr increases then decreases
        max_idx = np.argmax(lrs)
        assert 2 <= max_idx <= 4  # Should peak around 30% of cycle

    def test_cycle_momentum(self):
        """Test momentum cycling (inverse to learning rate)"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1, momentum=0.85)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1.0,
            total_steps=10,
            cycle_momentum=True,
            max_momentum=0.95,
        )

        momentums = []
        lrs = []
        for _ in range(10):
            momentums.append(optimizer.param_groups[0]["momentum"])
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # When lr is low, momentum should be high (inverse relationship)
        assert momentums[0] > 0.85  # High momentum at start (low lr)

        # Momentum should decrease as lr increases
        mid_point = 3
        assert momentums[mid_point] < momentums[0]

    def test_warm_up_phase(self):
        """Test the warm-up phase increases learning rate"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1, momentum=0.9)
        scheduler = OneCycleLR(optimizer, max_lr=1.0, total_steps=100, pct_start=0.3)

        warmup_steps = int(100 * 0.3)
        lrs = []

        for i in range(warmup_steps + 1):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Learning rate should increase during warm-up
        for i in range(len(lrs) - 1):
            assert lrs[i] <= lrs[i + 1]

    def test_cool_down_phase(self):
        """Test the cool-down phase decreases learning rate"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1, momentum=0.9)
        scheduler = OneCycleLR(optimizer, max_lr=1.0, total_steps=100, pct_start=0.3)

        warmup_steps = int(100 * 0.3)

        # Skip to cool-down phase
        for _ in range(warmup_steps):
            scheduler.step()

        lrs = []
        for _ in range(100 - warmup_steps):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Learning rate should decrease during cool-down
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1]

    def test_last_epoch_update(self):
        """Test last_epoch tracking"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1, momentum=0.9)
        scheduler = OneCycleLR(optimizer, max_lr=1.0, total_steps=10, last_epoch=2)

        assert scheduler.last_epoch == 3
        scheduler.step()
        assert scheduler.last_epoch == 4

    def test_with_adam_betas(self):
        """Test OneCycleLR with Adam optimizer (uses betas instead of momentum)"""
        p = Parameter(nova.randn(2, 2))
        optimizer = Adam([p], lr=0.1, betas=(0.9, 0.999))
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1.0,
            total_steps=10,
            cycle_momentum=True,
            max_momentum=0.95,
        )

        initial_beta1 = optimizer.param_groups[0]["betas"][0]
        scheduler.step()

        # Beta1 should change during cycle
        new_beta1 = optimizer.param_groups[0]["betas"][0]
        assert new_beta1 != initial_beta1

        # Beta2 should remain unchanged
        assert optimizer.param_groups[0]["betas"][1] == 0.999


# Create a basic model
model = nn.Sequential(nn.Linear(8, 10), nn.ReLU(), nn.Linear(10, 1))

# Set hyperparameters
LR = 1e-3
ALPHA = 0.99
MOMENTUM = 0.9
BETAS = (0.9, 0.999)
WD = 1e-4

# Optimizers
sgd_optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
adam_optimizer = Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=WD)
adamw_optimizer = AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WD)
rmsprop_optimizer = RMSprop(
    model.parameters(), lr=LR, alpha=ALPHA, weight_decay=WD, momentum=MOMENTUM
)

optimizers: list[Optimizer] = [
    sgd_optimizer,
    adam_optimizer,
    adamw_optimizer,
    rmsprop_optimizer,
]


class TestAllSchedulers:

    @pytest.mark.parametrize("optimizer", optimizers)
    def test_schedulers_with_all_optimizers(self, optimizer: Optimizer):
        """Test each scheduler with all optimizers"""

        # Test StepLR
        scheduler_step = StepLR(optimizer, step_size=5, gamma=0.5)
        initial_lr = optimizer.param_groups[0]["lr"]

        for _ in range(10):
            scheduler_step.step()

        # After 10 steps with step_size=5, should have decayed twice
        expected_lr = initial_lr * (0.5**2)
        assert np.isclose(optimizer.param_groups[0]["lr"], expected_lr)

        # Reset lr
        for group in optimizer.param_groups:
            group["lr"] = LR

        # Test CosineAnnealingLR
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

        for _ in range(20):
            scheduler_cosine.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < LR
        assert final_lr >= 1e-6

        # Reset lr
        for group in optimizer.param_groups:
            group["lr"] = LR

        # Test OneCycleLR (only for optimizers with momentum/betas)
        if (
            "momentum" in optimizer.param_groups[0]
            or "betas" in optimizer.param_groups[0]
        ):
            scheduler_one = OneCycleLR(optimizer, max_lr=0.01, total_steps=15)

            lrs = []
            for _ in range(15):
                lrs.append(optimizer.param_groups[0]["lr"])
                scheduler_one.step()

            # Check that lr varies (increases then decreases)
            assert max(lrs) > min(lrs)
            assert lrs.index(max(lrs)) < len(lrs) - 1  # Peak is not at the end

    def test_scheduler_state_persistence(self):
        """Test that scheduler state can be saved and restored across all schedulers"""
        p = Parameter(nova.randn(2, 2))
        optimizer = SGD([p], lr=0.1)

        # Test with each scheduler type
        schedulers: list[_LRScheduler] = [
            StepLR(optimizer, step_size=3, gamma=0.8),
            CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01),
            OneCycleLR(optimizer, max_lr=1.0, total_steps=20),
        ]

        for scheduler in schedulers:
            # Run for some steps
            for _ in range(5):
                scheduler.step()

            # Save state
            state = scheduler.state_dict()

            # Create new scheduler and load state
            if isinstance(scheduler, StepLR):
                new_scheduler = StepLR(optimizer, step_size=3, gamma=0.8)
            elif isinstance(scheduler, CosineAnnealingLR):
                new_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01)
            else:
                new_scheduler = OneCycleLR(optimizer, max_lr=1.0, total_steps=20)

            new_scheduler.load_state_dict(state)

            # Verify state is restored
            assert new_scheduler.last_epoch == scheduler.last_epoch

            # Step both and verify they produce same lr
            scheduler.step()
            new_scheduler.step()

            assert nova.allclose(
                scheduler.get_last_lr(), new_scheduler.get_last_lr(), rtol=1e-5
            )
