from __future__ import annotations
import nova
import numpy as np
from typing import TYPE_CHECKING, Literal
from ..metric import Metric

if TYPE_CHECKING:
    from nova import Tensor


class ROCAUC(Metric):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    ROC is a probability curve and AUC represents the degree or measure of separability.
    It tells how much the model is capable of distinguishing between classes.

    .. math::
        \\text{AUC} = \\int_{0}^{1} \\text{TPR}(\\text{FPR}) \\, d(\\text{FPR})

    Calculated using the trapezoidal rule approximation.

    Args:
        num_classes (int): Number of classes (currently supports binary classification best).

    Examples:
        >>> auc = ROCAUC()
        >>> # Probabilities for positive class
        >>> preds = nova.tensor([0.1, 0.4, 0.35, 0.8])
        >>> target = nova.tensor([0, 0, 1, 1])
        >>> auc.update(preds, target)
        >>> print(f"AUC: {auc.compute():.4f}")

    Note:
        - Unlike other metrics, ROCAUC must store **all** predictions and targets
          in memory to perform sorting and integration at the end.
        - Using this metric on extremely large datasets might consume high RAM.
        - Automatically detaches tensors to avoid memory leaks in the autograd graph.
        - For multi-class (num_classes > 2), this implementation treats `preds[:, 1]`
          as the positive class score by default.

    Reference:
        - ROC Curve: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """

    def __init__(
        self, num_classes: int = 2, task: Literal["multiclass", "binary"] = "multiclass"
    ):
        super().__init__()

        self.task = task
        if task == "binary" and num_classes != 2:
            raise ValueError("Binary task requires num_classes=2")

        self.preds_list: list[Tensor] = []
        self.target_list: list[Tensor] = []
        self.num_classes = num_classes

    def reset(self) -> None:
        """Clears the stored predictions and targets."""
        self.preds_list = []
        self.target_list = []

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Stores predictions and targets for later computation.

        Args:
            preds: Predictions tensor. If (N, C) and C=2, column 1 is selected.
            target: Ground truth tensor.
        """
        if preds.dim() > 1 and self.num_classes == 2:
            # Assume binary classification, take probability of positive class
            preds = preds[:, 1]

        # Detach to save memory (we don't need gradients for metrics)
        self.preds_list.append(preds.detach())
        self.target_list.append(target.detach())

    def compute(self) -> Tensor:
        """
        Computes the ROC AUC score using the trapezoidal rule.

        Returns:
            Tensor containing the AUC score (scalar).
        """
        y_pred = np.concatenate(
            [p.data for p in self.preds_list],
        )
        y_true = np.concatenate([t.data for t in self.target_list])

        # Sort by prediction score in descending order
        desc_score_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        y_pred_sorted = y_pred[desc_score_indices]

        # Find indices where thresholds change
        distinct_value_indices = np.where(np.diff(y_pred_sorted))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

        # Calculate Cumulative Sums for TPR and FPR
        tps = np.cumsum(y_true_sorted)[threshold_idxs]
        fps = (1 + threshold_idxs) - tps

        # Normalize
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        # Add (0,0) point
        tpr = np.r_[0, tpr]
        fpr = np.r_[0, fpr]

        # Calculate Area
        auc_score = np.trapezoid(tpr, fpr)

        return nova.tensor(auc_score)
