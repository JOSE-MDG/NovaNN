import nova
import pytest
import numpy as np
import nova.nn as nn
import nova.nn.functional as F
from nova import Tensor
from typing import Callable
from nova.utils import grad_check_wrt_inputs

# losses: mse_loss, l1_loss, smooth_l1_loss, bce_loss, bcewith_logits, nll_loss, cross_entropy, kl_div

nova.manual_seed(8)


class TestMSELoss:
    """Tests for Mean Squared Error Loss"""

    def test_mse_basic_calculation(self):
        """Test MSE with known values"""
        input = nova.tensor([2.5, 0.0, 2.0, 8.0])
        target = nova.tensor([3.0, -0.5, 2.0, 7.0])

        loss = F.mse_loss(input, target, reduction="mean")

        # Manual calculation: ((0.5)^2 + (0.5)^2 + (0)^2 + (1)^2) / 4 = 0.375
        expected = 0.375
        diff = abs(loss.item() - expected)
        assert diff < 1e-6, f"Expected {expected}, got {loss.item()}"

    def test_mse_reduction_none(self):
        """Test MSE with no reduction"""
        input = nova.tensor([1.0, 2.0, 3.0])
        target = nova.tensor([1.5, 2.5, 3.5])

        loss = F.mse_loss(input, target, reduction="none")

        assert (
            loss.shape == input.shape
        ), f"mismatch in shapes {loss.shape} != {input.shape}"
        expected = nova.tensor([0.25, 0.25, 0.25])
        assert nova.allclose(loss, expected), f"{str(loss)} != {str(expected)}"

    def test_mse_reduction_sum(self):
        """Test MSE with sum reduction"""
        input = nova.tensor([1.0, 2.0, 3.0])
        target = nova.tensor([1.5, 2.5, 3.5])

        loss = F.mse_loss(input, target, reduction="sum")

        expected = 0.75  # 0.25 * 3
        diff = abs(loss.item() - expected)
        assert diff < 1e-6, f"{diff} > 1e-6"

    def test_mse_with_weights(self):
        """Test MSE with element-wise weights"""
        input = nova.tensor([1.0, 2.0, 3.0])
        target = nova.tensor([1.5, 2.5, 3.5])
        weight = nova.tensor([1.0, 2.0, 3.0])

        loss = F.mse_loss(input, target, weight=weight, reduction="mean")

        # (0.25*1 + 0.25*2 + 0.25*3) / 3 = 0.5
        expected = 0.5
        diff = abs(loss.item() - expected)
        assert diff < 1e-6, f"{diff} > 1e-6"

    def test_mse_weight_shape_mismatch(self):
        """Test that mismatched weight shape raises error"""
        input = nova.tensor([1.0, 2.0, 3.0])
        target = nova.tensor([1.5, 2.5, 3.5])
        weight = nova.tensor([1.0, 2.0])

        with pytest.raises(ValueError, match="same shape"):
            F.mse_loss(input, target, weight=weight)

    def test_mse_gradient(self):
        """Test MSE gradient computation"""
        input = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = nova.tensor([1.5, 2.5, 3.5])

        criterion = nn.MSELoss(reduction="mean")

        loss = criterion(input, target)
        loss.backward()

        # Gradient: 2 * (input - target) / n = 2 * [-0.5, -0.5, -0.5] / 3
        expected_grad = nova.tensor([-0.333333, -0.333333, -0.333333])
        assert input.grad is not None
        assert nova.allclose(input.grad, expected_grad, atol=1e-5)

        # Reset gradients to None
        input.zero_grad()
        assert input.grad is None

        # Numerical Gradient:
        analitycal, numerical = grad_check_wrt_inputs(criterion, input, target)

        assert nova.allclose(analitycal[0], numerical[0], rtol=3e-3, atol=5e-3)

    def test_mse_multidimensional(self):
        """Test MSE with multidimensional tensors"""
        input = nova.randn(3, 5, 4)
        target = nova.randn(3, 5, 4)

        loss = F.mse_loss(input, target, reduction="mean")

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # MSE is always non-negative

    def test_mse_zero_loss(self):
        """Test MSE when input equals target"""
        input = nova.tensor([1.0, 2.0, 3.0])
        target = nova.tensor([1.0, 2.0, 3.0])

        loss = F.mse_loss(input, target)
        diff = abs(loss.item())
        assert diff < 1e-7, f"{diff} > 1e-7"


class TestL1Loss:
    """Tests for L1 (Mean Absolute Error) Loss"""

    def test_l1_basic_calculation(self):
        """Test L1 with known values"""
        input = nova.tensor([2.5, 0.0, 2.0, 8.0])
        target = nova.tensor([3.0, -0.5, 2.0, 7.0])

        loss = F.l1_loss(input, target, reduction="mean")

        # |0.5| + |0.5| + |0| + |1| = 2.0 / 4 = 0.5
        expected = 0.5
        diff = abs(loss.item() - expected)
        assert diff < 1e-6, f"{diff} < 1e-6"

    def test_l1_reduction_modes(self):
        """Test L1 with different reduction modes"""
        input = nova.tensor([1.0, 3.0, 5.0])
        target = nova.tensor([2.0, 4.0, 6.0])

        loss_none = F.l1_loss(input, target, reduction="none")
        loss_mean = F.l1_loss(input, target, reduction="mean")
        loss_sum = F.l1_loss(input, target, reduction="sum")

        assert loss_none.shape == input.shape
        assert nova.allclose(loss_none, nova.tensor([1.0, 1.0, 1.0]))
        assert (
            abs(loss_mean.item() - 1.0) < 1e-6
        ), f"diff: {abs(loss_mean.item() - 1.0)} > 1e-6"
        assert (
            abs(loss_sum.item() - 3.0) < 1e-6
        ), f"diff: {abs(loss_sum.item() - 3.0)} > 1e-6"

    def test_l1_gradient(self):
        """Test L1 gradient computation"""
        input = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = nova.tensor([1.5, 1.5, 3.5])

        criterion = nn.L1Loss(reduction="mean")
        loss = criterion(input, target)
        loss.backward()

        # Gradient: sign(input - target) / n
        # [-1, 1, -1] / 3
        expected_grad = nova.tensor([-0.333333, 0.333333, -0.333333])
        assert input.grad is not None
        assert nova.allclose(input.grad, expected_grad, atol=1e-5)

        # Reset gradients to None
        input.zero_grad()

        # Numerical Gradient:
        analitycal, numerical = grad_check_wrt_inputs(criterion, input, target)

        assert nova.allclose(analitycal[0], numerical[0], rtol=3e-3, atol=5e-3)


class TestSmoothL1Loss:
    """Tests for Smooth L1 (Huber) Loss"""

    def test_smooth_l1_basic_calculation(self):
        """Test Smooth L1 with known values"""
        input = nova.tensor([2.5, 0.0, 2.0, 8.0])
        target = nova.tensor([3.0, -0.5, 2.0, 7.0])

        loss = F.smooth_l1_loss(input, target, beta=1.0)

        # For beta=1.0:
        # |diff| < 1: 0.5 * diff^2 / 1 -> [0.125, 0.125, 0, ...]
        # |diff| >= 1: |diff| - 0.5 -> [..., 0.5]
        expected = 0.1875  # (0.125 + 0.125 + 0 + 0.5) / 4
        diff = abs(loss.item() - expected)
        assert diff < 1e-6, f"{diff} > 1e-6"

    def test_smooth_l1_beta_variations(self):
        """Test Smooth L1 with different beta values"""
        input = nova.tensor([0.0, 2.0])
        target = nova.tensor([1.0, 3.0])

        loss_beta_1 = F.smooth_l1_loss(input, target, beta=1.0, reduction="none")
        loss_beta_2 = F.smooth_l1_loss(input, target, beta=2.0, reduction="none")

        # Smaller beta -> more L1-like
        # Larger beta -> more L2-like
        assert loss_beta_1.shape == input.shape
        assert loss_beta_2.shape == input.shape

    def test_smooth_l1_convergence_to_l1(self):
        """Test that Smooth L1 converges to L1 for large errors"""
        input = nova.tensor([0.0, 10.0])
        target = nova.tensor([5.0, 15.0])
        beta = 1.0

        smooth_l1 = F.smooth_l1_loss(input, target, beta=beta, reduction="none")
        l1 = F.l1_loss(input, target, reduction="none")

        # For large errors, smooth_l1 ≈ l1 - 0.5*beta
        assert abs(smooth_l1[0].item() - (l1[0].item() - 0.5 * beta)) < 1e-5
        assert abs(smooth_l1[1].item() - (l1[1].item() - 0.5 * beta)) < 1e-5


class TestBinaryCrossEntropy:
    """Tests for Binary Cross Entropy Loss"""

    def test_bce_basic_calculation(self):
        """Test BCE with known values"""
        input = nova.tensor([0.8, 0.3, 0.9, 0.2])
        target = nova.tensor([1.0, 0.0, 1.0, 0.0])

        loss = F.binary_cross_entropy(input, target, reduction="mean")

        # Manual: -(1*log(0.8) + 0*log(0.2) + 1*log(0.9) + 0*log(0.8)) / 4
        assert loss.item() > 0
        assert loss.item() < 1  # Reasonable range

    def test_bce_perfect_prediction(self):
        """Test BCE with perfect predictions"""
        input = nova.tensor([1.0, 0.0, 1.0, 0.0])
        target = nova.tensor([1.0, 0.0, 1.0, 0.0])

        loss = F.binary_cross_entropy(input, target)

        # Should be very close to 0 (clamped to avoid log(0))
        assert loss.item() < 1e-10

    def test_bce_numerical_stability(self):
        """Test BCE doesn't produce NaN or Inf"""
        # Extreme values that could cause numerical issues
        input = nova.tensor([0.99999, 0.00001, 0.5])
        target = nova.tensor([1.0, 0.0, 1.0])

        loss = F.binary_cross_entropy(input, target)

        assert not nova.any(nova.isnan(loss))
        assert not nova.any(nova.isinf(loss))

    def test_bce_gradient(self):
        """Test BCE gradient computation with safe values"""
        input = nova.tensor([0.6, 0.4, 0.8], requires_grad=True)
        target = nova.tensor([1.0, 0.0, 1.0])

        criterion = nn.BCELoss()
        loss = criterion(input, target)
        loss.backward()

        assert input.grad is not None
        assert input.grad.shape == input.shape

        assert np.isfinite(input.grad).all()
        assert input.grad[0].item() < 0  # target=1
        assert input.grad[1].item() > 0  # target=0
        assert input.grad[2].item() < 0  # target=1


class TestBCEWithLogits:
    """Tests for Binary Cross Entropy with Logits Loss"""

    def test_bce_with_logits_basic(self):
        """Test BCE with logits basic calculation"""
        logits = nova.tensor([1.5, -0.5, 2.0, -1.0])
        target = nova.tensor([1.0, 0.0, 1.0, 0.0])

        loss = F.binary_cross_entropy_with_logits(logits, target)

        assert loss.item() > 0
        assert not nova.isnan(loss)

    def test_bce_with_logits_vs_bce(self):
        """Test equivalence with sigmoid + BCE"""
        logits = nova.tensor([1.5, -0.5, 2.0, -1.0])
        target = nova.tensor([1.0, 0.0, 1.0, 0.0])

        loss_with_logits = F.binary_cross_entropy_with_logits(logits, target)

        probs = F.sigmoid(logits)
        loss_bce = F.binary_cross_entropy(probs, target)

        # Should be approximately equal
        assert abs(loss_with_logits.item() - loss_bce.item()) < 1e-5

    def test_bce_with_logits_pos_weight(self):
        """Test BCE with positive class weighting"""
        logits = nova.tensor([1.0, -1.0, 0.5, -0.5])
        target = nova.tensor([1.0, 0.0, 1.0, 0.0])
        pos_weight = nova.tensor([2.0])

        loss_weighted = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
        )
        loss_normal = F.binary_cross_entropy_with_logits(logits, target)

        # Weighted loss should be different
        assert loss_weighted.item() != loss_normal.item()

    def test_bce_with_logits_extreme_values(self):
        """Test numerical stability with extreme logits"""
        logits = nova.tensor([100.0, -100.0, 0.0])
        target = nova.tensor([1.0, 0.0, 1.0])

        loss = F.binary_cross_entropy_with_logits(logits, target)

        assert not nova.isnan(loss)
        assert not nova.isinf(loss)
        assert loss.item() > 0


class TestNLLLoss:
    """Tests for Negative Log Likelihood Loss"""

    def test_nll_basic_calculation(self):
        """Test NLL with known values"""
        log_probs = F.log_softmax(nova.tensor([[2.0, 1.0, 0.1]]), dim=1)
        target = nova.tensor([0], dtype=nova.long)

        loss = F.nll_loss(log_probs, target)

        # Should extract log_prob of class 0
        expected = -log_probs[0, 0].item()
        diff = abs(loss.item() - expected)
        assert diff < 1e-6, f"{diff} > 1e-6"

    def test_nll_batch(self):
        """Test NLL with batch of samples"""
        log_probs = F.log_softmax(
            nova.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3], [1.0, 1.5, 2.0]]), dim=1
        )
        target = nova.tensor([0, 1, 2], dtype=nova.long)

        loss = F.nll_loss(log_probs, target)

        # Mean of negative log probs at target indices
        manual_loss = -(log_probs[0, 0] + log_probs[1, 1] + log_probs[2, 2]) / 3
        diff = abs(loss.item() - manual_loss.item())
        assert diff < 1e-6, f"{diff} > 1e-6"

    def test_nll_with_class_weights(self):
        """Test NLL with per-class weights"""
        log_probs = F.log_softmax(
            nova.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]), dim=1
        )
        target = nova.tensor([0, 1], dtype=nova.long)
        weight = nova.tensor([0.5, 2.0, 1.0])

        loss_weighted = F.nll_loss(log_probs, target, weight=weight)
        loss_normal = F.nll_loss(log_probs, target)

        assert (
            loss_weighted.item() != loss_normal.item()
        ), f"weighted loss is different to normal loss {loss_weighted.item()} != {loss_normal.item()}"

    def test_nll_reduction_modes(self):
        """Test NLL with different reduction modes"""
        log_probs = F.log_softmax(nova.tensor([[2.0, 1.0, 0.1]]), dim=1)
        target = nova.tensor([0], dtype=nova.long)

        loss_none = F.nll_loss(log_probs, target, reduction="none")
        loss_mean = F.nll_loss(log_probs, target, reduction="mean")
        loss_sum = F.nll_loss(log_probs, target, reduction="sum")

        assert loss_none.shape == (1,)
        assert loss_mean.ndim == 0
        assert loss_sum.ndim == 0


class TestCrossEntropy:
    """Tests for Cross Entropy Loss"""

    def test_cross_entropy_basic(self):
        """Test cross entropy basic calculation"""
        logits = nova.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        target = nova.tensor([0, 1], dtype=nova.long)

        loss = F.cross_entropy(logits, target)

        assert loss.item() > 0
        assert not nova.isnan(loss)

    def test_cross_entropy_equivalence(self):
        """Test cross entropy equals log_softmax + nll_loss"""
        logits = nova.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        target = nova.tensor([0, 1], dtype=nova.long)

        ce_loss = F.cross_entropy(logits, target)

        log_probs = F.log_softmax(logits, dim=1)
        nll_loss = F.nll_loss(log_probs, target)

        assert (
            abs(ce_loss.item() - nll_loss.item()) < 1e-6
        ), f"There isn't equivalence from cross entropy to nllloss"

    def test_cross_entropy_with_weights(self):
        """Test cross entropy with class weights"""
        logits = nova.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        target = nova.tensor([0, 1], dtype=nova.long)
        weight = nova.tensor([0.5, 2.0, 1.0])

        loss_weighted = F.cross_entropy(logits, target, weight=weight)
        loss_normal = F.cross_entropy(logits, target)

        assert (
            loss_weighted.item() != loss_normal.item()
        ), f"weighted loss is different to normal loss {loss_weighted.item()} != {loss_normal.item()}"

    def test_cross_entropy_gradient(self):
        """Test cross entropy gradient"""
        logits = nova.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True)
        target = nova.tensor([0, 1], dtype=nova.long)

        criterion = nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(logits, target)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

        logits.zero_grad()
        assert logits.grad is None
        analitycal, numerical = grad_check_wrt_inputs(criterion, logits, target)
        assert nova.allclose(analitycal[0], numerical[0], rtol=3e-3, atol=5e-3)

    def test_cross_entropy_perfect_prediction(self):
        """Test cross entropy with perfect predictions"""
        # Very high logit for correct class
        logits = nova.tensor([[100.0, 0.0, 0.0]])
        target = nova.tensor([0], dtype=nova.long)

        loss = F.cross_entropy(logits, target)

        # Should be very small (but not exactly 0 due to numerical precision)
        assert loss.item() < 0.01


class TestKLDivLoss:
    """Tests for Kullback-Leibler Divergence Loss"""

    def test_kl_div_basic(self):
        """Test KL divergence basic calculation"""
        log_probs = F.log_softmax(nova.tensor([[2.0, 1.0, 0.5]]), dim=1)
        target = nova.tensor([[0.7, 0.2, 0.1]])

        loss = F.kl_div(log_probs, target, reduction="batchmean")

        assert loss.item() >= 0  # KL divergence is non-negative

    def test_kl_div_identical_distributions(self):
        """Test KL divergence is zero for identical distributions"""
        probs = nova.tensor([[0.3, 0.5, 0.2]])
        log_probs = nova.log(probs)

        loss = F.kl_div(log_probs, probs, reduction="batchmean")

        # Should be very close to 0
        assert abs(loss.item()) < 1e-5

    def test_kl_div_log_target(self):
        """Test KL divergence with log target"""
        log_probs = F.log_softmax(nova.tensor([[2.0, 1.0, 0.5]]), dim=1)
        log_target = F.log_softmax(nova.tensor([[1.5, 1.2, 0.8]]), dim=1)

        loss = F.kl_div(log_probs, log_target, log_target=True, reduction="batchmean")

        assert loss.item() >= 0
        assert not nova.isnan(loss)

    def test_kl_div_reduction_modes(self):
        """Test KL divergence with different reduction modes"""
        log_probs = F.log_softmax(nova.randn(4, 5), dim=1)
        target = F.softmax(nova.randn(4, 5), dim=1)

        loss_none = F.kl_div(log_probs, target, reduction="none")
        loss_mean = F.kl_div(log_probs, target, reduction="mean")
        loss_sum = F.kl_div(log_probs, target, reduction="sum")
        loss_batchmean = F.kl_div(log_probs, target, reduction="batchmean")

        # Check shapes
        assert loss_none.shape == (4,)  # Sum over features, not batch
        assert loss_mean.ndim == 0
        assert loss_sum.ndim == 0
        assert loss_batchmean.ndim == 0

        # batchmean should be sum / batch_size
        assert abs(loss_batchmean.item() - (loss_sum.item() / 4)) < 1e-5


class TestReduceFunction:
    """Tests for the _reduce helper function"""

    def test_reduce_none(self):
        """Test no reduction"""
        from nova.nn.functional import _reduce

        loss = nova.tensor([1.0, 2.0, 3.0])
        reduced = _reduce(loss, "none")

        assert nova.allclose(reduced, loss)

    def test_reduce_sum(self):
        """Test sum reduction"""
        from nova.nn.functional import _reduce

        loss = nova.tensor([1.0, 2.0, 3.0])
        reduced = _reduce(loss, "sum")

        assert abs(reduced.item() - 6.0) < 1e-6

    def test_reduce_mean(self):
        """Test mean reduction"""
        from nova.nn.functional import _reduce

        loss = nova.tensor([1.0, 2.0, 3.0])
        reduced = _reduce(loss, "mean")

        assert abs(reduced.item() - 2.0) < 1e-6

    def test_reduce_batchmean(self):
        """Test batchmean reduction"""
        from nova.nn.functional import _reduce

        loss = nova.tensor([1.0, 2.0, 3.0, 4.0])
        reduced = _reduce(loss, "batchmean", batch_size=2)

        # Sum / batch_size = 10 / 2 = 5.0
        assert abs(reduced.item() - 5.0) < 1e-6

    def test_reduce_invalid_mode(self):
        """Test invalid reduction mode raises error"""
        from nova.nn.functional import _reduce

        loss = nova.tensor([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="reduction expect"):
            _reduce(loss, "invalid")

    def test_reduce_batchmean_no_batch_size(self):
        """Test batchmean without batch_size raises error"""
        from nova.nn.functional import _reduce

        loss = nova.tensor([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="batch size must be specified"):
            _reduce(loss, "batchmean")


class TestLossEdgeCases:
    """Tests for edge cases across all loss functions"""

    def test_empty_tensor(self):
        """Test losses handle empty tensors appropriately"""
        # TODO: This might raise or return a specific value
        pass

    def test_single_element(self):
        """Test losses with single element tensors"""
        input = nova.tensor([2.0])
        target = nova.tensor([3.0])

        loss = F.mse_loss(input, target)
        assert abs(loss.item() - 1.0) < 1e-6

    def test_large_batch(self):
        """Test losses scale properly with large batches"""
        input = nova.randn(1000, 10)
        target = nova.randn(1000, 10)

        loss = F.mse_loss(input, target)

        assert not nova.isnan(loss)
        assert not nova.isinf(loss)

    def test_mixed_dtypes(self):
        """Test losses handle mixed dtypes"""
        pass


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_all_losses_reduction_modes(reduction):
    """Test all loss functions support all reduction modes"""
    input = nova.randn(3, 5)
    target = nova.randn(3, 5)

    losses_to_test: list[tuple[Callable[[Tensor, Tensor], Tensor], dict]] = [
        (F.mse_loss, {}),
        (F.l1_loss, {}),
        (F.smooth_l1_loss, {"beta": 1.0}),
    ]

    for loss_fn, kwargs in losses_to_test:
        loss = loss_fn(input, target, reduction=reduction, **kwargs)

        if reduction == "none":
            assert loss.shape == input.shape
        else:
            assert loss.ndim == 0  # Scalar
