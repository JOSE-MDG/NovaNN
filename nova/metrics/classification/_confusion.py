from __future__ import annotations
import nova
import numpy as np
from typing import TYPE_CHECKING
from ..metric import Metric

if TYPE_CHECKING:
    from nova import Tensor


class ConfusionMatrix(Metric):
    """
    Computes the multi-class Confusion Matrix.

    The confusion matrix is a table that is often used to describe the performance
    of a classification model. Each row of the matrix represents the instances in
    a predicted class while each column represents the instances in an actual class
    (or vice-versa).

    .. math::
        C_{ij} = \\text{count}(y_{true} == i, y_{pred} == j)

    Where:
    - $i$ is the true class index
    - $j$ is the predicted class index
    - Diagonal elements $C_{ii}$ represent correct predictions (True Positives)
    - Off-diagonal elements represent errors

    Examples:
        >>> # 3-class classification
        >>> cm = ConfusionMatrix(num_classes=3)
        >>> preds = nova.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]]) # Class 1, Class 0
        >>> target = nova.tensor([1, 0])
        >>> cm.update(preds, target)
        >>> print(cm.compute())
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.]])

        >>> # Training loop
        >>> cm = ConfusionMatrix(num_classes=10)
        >>> for x, y in val_loader:
        ...     preds = model(x)
        ...     cm.update(preds, y)
        >>> matrix = cm.compute()
        >>> # Visualize with matplotlib/seaborn...

    Note:
        - Uses efficient numpy histogram calculation internally
        - Supports both probabilities/logits (automatically applies argmax) and class indices
        - Row index = True Class, Column index = Predicted Class
        - Returns a Float Tensor
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.matrix = None
        super().__init__()

    def reset(self) -> None:
        """Resets the confusion matrix to zeros."""
        self.matrix = nova.zeros((self.num_classes, self.num_classes))

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Accumulates predictions into the confusion matrix.

        Args:
            preds: (N, C) tensor of logits/probs or (N,) tensor of indices.
            target: (N,) tensor of ground truth class indices.

        Note:
            If preds has shape (N, C), argmax(dim=1) is applied automatically.
        """
        if preds.ndim > 1:
            preds = preds.argmax(dim=1)

        preds = preds.reshape(-1)
        target = target.reshape(-1)

        p_data = preds.data
        t_data = target.data
        H, _, _ = np.histogram2d(
            t_data,
            p_data,
            bins=self.num_classes,
            range=[[0, self.num_classes], [0, self.num_classes]],
        )

        self.matrix += nova.tensor(H)

    def compute(self) -> Tensor:
        """
        Returns the accumulated confusion matrix.

        Returns:
            Tensor of shape (num_classes, num_classes).
        """
        return self.matrix
