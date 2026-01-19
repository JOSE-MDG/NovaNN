from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal
from ..metric import Metric
from ._confusion import ConfusionMatrix

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Average


class ClassificationStat(Metric):
    """
    Base class for statistics derived from the confusion matrix.

    Handles the calculation of TP, FP, TN, FN and supports different averaging
    strategies for multi-class problems.

    Args:
        num_classes (int): Number of classes.
        average (Average): Strategy to reduce results across classes:
            - 'micro': Calculate metrics globally by counting total TP, FN and FP.
            - 'macro': Calculate metrics for each label, and find their unweighted mean.
              Does not take label imbalance into account.
            - 'weighted': Calculate metrics for each label, and find their average weighted
              by support (the number of true instances for each label).
            - None: Return the score for each class separately.

    Raises:
        ValueError: If `average` is not one of the allowed values.
    """

    def __init__(
        self,
        num_classes: int,
        average: Average = "macro",
        task: Literal["multiclass", "binary"] = "multiclass",
    ) -> None:

        self.task = task

        if task == "binary" and num_classes != 2:
            raise ValueError("Binary task requires num_classes=2")

        self.num_classes = num_classes
        self.cm = ConfusionMatrix(num_classes)
        self.average: Average = average
        self._check_avg(average)
        super().__init__()

    def reset(self) -> None:
        """Resets the internal confusion matrix."""
        self.cm.reset()

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Updates the internal confusion matrix with new predictions."""
        self.cm.update(preds, target)

    @abstractmethod
    def _calculate(self, tp, fp, fn, tn, support) -> Tensor:
        pass

    def compute(self) -> Tensor:
        """
        Computes the final metric based on the accumulated confusion matrix.

        Returns:
            Tensor: scalar if average is set, or (num_classes,) if average is None.
        """
        matrix = self.cm.compute()  # (num_classes, num_classes)

        tp = matrix.diag()

        support = matrix.sum(dim=1)

        fp = matrix.sum(dim=0) - tp

        fn = support - tp

        tn = matrix.sum() - (tp + fp + fn)

        if self.average == "micro":
            return self._calculate(
                tp.sum(), fp.sum(), fn.sum(), tn.sum(), support.sum()
            )

        scores = self._calculate(tp, fp, fn, tn, support)

        if self.average == "macro":
            return scores.mean()
        elif self.average == "weighted":
            return (scores * support).sum() / support.sum()
        else:
            return scores

    def _check_avg(self, avg: Average) -> None:
        avgs = ("micro", "macro", "weighted", None)
        if avg not in avgs:
            raise ValueError(
                f"average expect ('micro', 'macro', 'weighted', None), got {avg}"
            )


class Accuracy(ClassificationStat):
    """
    Computes Accuracy classification score.

    .. math::
        \\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}

    Examples:
        >>> acc = Accuracy(num_classes=2)
        >>> preds = nova.tensor([[0.9, 0.1], [0.2, 0.8]])
        >>> target = nova.tensor([0, 1])
        >>> acc.update(preds, target)
        >>> print(acc.compute())
    """

    def __init__(
        self,
        num_classes,
        average="micro",
        task: Literal["multiclass", "binary"] = "multiclass",
    ):
        super().__init__(num_classes, average, task)

    def _calculate(self, tp, fp, fn, tn, support) -> Tensor:
        # If this is micro-average (tp, fp, fn are scalars)
        if tp.ndim == 0:
            # Micro: just TP / Total samples
            return tp / (support + 1e-8)
        else:
            # Per-class: tp / support for that class
            return tp / (support + 1e-8)


class Precision(ClassificationStat):
    """
    Computes Precision (also known as Positive Predictive Value).

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives.

    .. math::
        \\text{Precision} = \\frac{TP}{TP + FP}

    Interpretation:
        - "Of all the samples predicted as Positive, how many were actually Positive?"
        - High precision means low False Positive rate.
    """

    def _calculate(self, tp, fp, fn, tn, support):
        return tp / (tp + fp + 1e-8)


class Recall(ClassificationStat):
    """
    Computes Recall (also known as Sensitivity or True Positive Rate).

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives.

    .. math::
        \\text{Recall} = \\frac{TP}{TP + FN}

    Interpretation:
        - "Of all the samples that were actually Positive, how many did we find?"
        - High recall means low False Negative rate.
    """

    def _calculate(self, tp, fp, fn, tn, support):
        return tp / (tp + fn + 1e-8)


class F1Score(ClassificationStat):
    """
    Computes F1 Score (Harmonic mean of Precision and Recall).

    The F1 score can be interpreted as a harmonic mean of the precision and recall,
    where an F1 score reaches its best value at 1 and worst score at 0.

    .. math::
        F_1 = 2 \\cdot \\frac{\\text{precision} \\cdot \\text{recall}}{\\text{precision} + \\text{recall}}

    Examples:
        >>> f1 = F1Score(num_classes=3, average='macro')
        >>> # ... update ...
        >>> print(f"F1: {f1.compute():.4f}")
    """

    def _calculate(self, tp, fp, fn, tn, support):
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        return 2 * (prec * rec) / (prec + rec + 1e-8)
