import pytest
import nova
import numpy as np
from nova.metrics import (
    MSE,
    MAE,
    R2Score,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ROCAUC,
    ConfusionMatrix,
)


class TestMSE:

    def test_mse_basic(self):
        """Test basic MSE calculation"""
        mse = MSE(squared=True)
        preds = nova.tensor([1.0, 2.0, 3.0])
        targets = nova.tensor([1.0, 2.0, 3.0])

        mse.update(preds, targets)
        result = mse.compute()

        assert np.isclose(result.item(), 0.0)

    def test_mse_with_error(self):
        """Test MSE with actual errors"""
        mse = MSE(squared=True)
        preds = nova.tensor([2.5, 0.0, 2.0])
        targets = nova.tensor([3.0, -0.5, 2.0])

        mse.update(preds, targets)
        result = mse.compute()

        # (0.5^2 + 0.5^2 + 0^2) / 3 = 0.5/3 = 0.1666...
        assert np.isclose(result.item(), 0.1666666, atol=1e-5)

    def test_rmse(self):
        """Test RMSE (squared=False)"""
        rmse = MSE(squared=False)
        preds = nova.tensor([3.0, -0.5, 2.0])
        targets = nova.tensor([2.5, 0.0, 2.0])

        rmse.update(preds, targets)
        result = rmse.compute()

        # sqrt(0.16666) ≈ 0.408
        assert np.isclose(result.item(), 0.408, atol=1e-2)

    def test_mse_reset(self):
        """Test MSE reset functionality"""
        mse = MSE()
        preds1 = nova.tensor([1.0, 2.0])
        targets1 = nova.tensor([2.0, 3.0])

        mse.update(preds1, targets1)
        mse.reset()

        preds2 = nova.tensor([1.0])
        targets2 = nova.tensor([1.0])
        mse.update(preds2, targets2)

        result = mse.compute()
        assert np.isclose(result.item(), 0.0)

    def test_mse_multiple_batches(self):
        """Test MSE across multiple batches"""
        mse = MSE()

        # Batch 1
        mse.update(nova.tensor([1.0, 2.0]), nova.tensor([1.0, 2.0]))
        # Batch 2
        mse.update(nova.tensor([3.0, 4.0]), nova.tensor([3.0, 5.0]))

        result = mse.compute()
        # Error only in last element: 1^2 / 4 = 0.25
        assert np.isclose(result.item(), 0.25)


class TestMAE:

    def test_mae_basic(self):
        """Test basic MAE calculation"""
        mae = MAE()
        preds = nova.tensor([1.0, 2.0, 3.0])
        targets = nova.tensor([1.0, 2.0, 3.0])

        mae.update(preds, targets)
        result = mae.compute()

        assert np.isclose(result.item(), 0.0)

    def test_mae_with_error(self):
        """Test MAE with actual errors"""
        mae = MAE()
        preds = nova.tensor([2.5, 0.0, 2.0])
        targets = nova.tensor([3.0, -0.5, 2.0])

        mae.update(preds, targets)
        result = mae.compute()

        # (0.5 + 0.5 + 0) / 3 = 1/3 = 0.333...
        assert np.isclose(result.item(), 0.3333, atol=1e-3)

    def test_mae_robust_to_outliers(self):
        """Test MAE is more robust than MSE to outliers"""
        mae = MAE()
        mse = MSE()

        preds = nova.tensor([1.0, 2.0, 10.0])
        targets = nova.tensor([1.1, 2.1, 3.0])

        mae.update(preds, targets)
        mse.update(preds, targets)

        mae_result = mae.compute().item()
        mse_result = mse.compute().item()

        # MSE should be much larger due to outlier
        assert mse_result > mae_result


class TestR2Score:

    def test_r2_perfect_fit(self):
        """Test R² = 1 for perfect predictions"""
        r2 = R2Score()
        preds = nova.tensor([1.0, 2.0, 3.0, 4.0])
        targets = nova.tensor([1.0, 2.0, 3.0, 4.0])

        r2.update(preds, targets)
        result = r2.compute()

        assert np.isclose(result.item(), 1.0)

    def test_r2_baseline(self):
        """Test R² ≈ 0 for mean prediction"""
        r2 = R2Score()
        targets = nova.tensor([1.0, 2.0, 3.0, 4.0])
        mean_pred = targets.mean()
        preds = nova.full_like(targets, mean_pred.item())

        r2.update(preds, targets)
        result = r2.compute()

        # Should be close to 0
        assert np.isclose(result.item(), 0.0, atol=1e-6)

    def test_r2_good_fit(self):
        """Test R² for a good (but not perfect) fit"""
        r2 = R2Score()
        preds = nova.tensor([3.0, 2.5, 4.0, 5.5])
        targets = nova.tensor([3.2, 2.4, 4.1, 5.0])

        r2.update(preds, targets)
        result = r2.compute()

        # Should be close to 1
        assert result.item() > 0.9


class TestAccuracy:

    def test_accuracy_perfect(self):
        """Test 100% accuracy"""
        acc = Accuracy(num_classes=3, average="micro")
        preds = nova.tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        targets = nova.tensor([0, 1, 2])

        acc.update(preds, targets)
        result = acc.compute()

        assert np.isclose(result.item(), 1.0)

    def test_accuracy_partial(self):
        """Test partial accuracy"""
        acc = Accuracy(num_classes=2, average="micro")
        preds = nova.tensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
        targets = nova.tensor([0, 1, 1, 0])  # 2 correct out of 4

        acc.update(preds, targets)
        result = acc.compute()

        assert np.isclose(result.item(), 0.5)

    def test_accuracy_zero(self):
        """Test 0% accuracy"""
        acc = Accuracy(num_classes=2, average="micro")
        preds = nova.tensor([[0.9, 0.1], [0.9, 0.1]])
        targets = nova.tensor([1, 1])  # All wrong

        acc.update(preds, targets)
        result = acc.compute()

        assert np.isclose(result.item(), 0.0)


class TestPrecisionRecallF1:

    def test_precision_perfect(self):
        """Test perfect precision"""
        prec = Precision(num_classes=2, average="macro")
        preds = nova.tensor([[0.9, 0.1], [0.1, 0.9]])
        targets = nova.tensor([0, 1])

        prec.update(preds, targets)
        result = prec.compute()

        assert np.isclose(result.item(), 1.0)

    def test_recall_perfect(self):
        """Test perfect recall"""
        rec = Recall(num_classes=2, average="macro")
        preds = nova.tensor([[0.9, 0.1], [0.1, 0.9]])
        targets = nova.tensor([0, 1])

        rec.update(preds, targets)
        result = rec.compute()

        assert np.isclose(result.item(), 1.0)

    def test_f1_score(self):
        """Test F1 score"""
        f1 = F1Score(num_classes=2, average="macro")
        preds = nova.tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
        targets = nova.tensor([0, 1, 1])  # One FP for class 0

        f1.update(preds, targets)
        result = f1.compute()

        # F1 should be between 0 and 1
        assert 0 <= result.item() <= 1


class TestConfusionMatrix:

    def test_confusion_matrix_binary(self):
        """Test binary confusion matrix"""
        cm = ConfusionMatrix(num_classes=2)
        preds = nova.tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        targets = nova.tensor([0, 1, 0, 0])  # Last one is wrong

        cm.update(preds, targets)
        matrix = cm.compute()

        # Expected:
        # [[2, 1],
        #  [0, 1]]

        assert matrix[0, 0].item() == 2.0
        assert matrix[1, 1].item() == 1.0
        assert matrix[1, 0].item() == 0.0

    def test_confusion_matrix_multiclass(self):
        """Test multiclass confusion matrix"""
        cm = ConfusionMatrix(num_classes=3)
        preds = nova.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        targets = nova.tensor([0, 1, 2])

        cm.update(preds, targets)
        matrix = cm.compute()

        # Perfect predictions -> diagonal matrix
        assert matrix[0, 0].item() == 1
        assert matrix[1, 1].item() == 1
        assert matrix[2, 2].item() == 1
        assert matrix.sum().item() == 3


class TestROCAUC:

    def test_roc_auc_perfect(self):
        """Test ROC AUC = 1 for perfect predictions"""
        auc = ROCAUC(num_classes=2)
        preds = nova.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        targets = nova.tensor([1, 1, 0, 0])

        auc.update(preds, targets)
        result = auc.compute()

        assert np.isclose(result.item(), 1.0)

    def test_roc_auc_random(self):
        """Test ROC AUC ≈ 0.5 for random predictions"""
        auc = ROCAUC(num_classes=2)
        # Random-ish predictions
        preds = nova.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        targets = nova.tensor([1, 0, 1, 0])

        auc.update(preds, targets)
        result = auc.compute()

        # Should be around 0.5
        assert 0.4 <= result.item() <= 0.6

    def test_roc_auc_good_separator(self):
        """Test ROC AUC for good separation"""
        auc = ROCAUC(num_classes=2)
        preds = nova.tensor([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])
        targets = nova.tensor([1, 1, 0, 0])

        auc.update(preds, targets)
        result = auc.compute()

        # Good separation
        assert result.item() > 0.9
