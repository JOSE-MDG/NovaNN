from __future__ import annotations
import nova
from typing import TYPE_CHECKING
from ..metric import Metric

if TYPE_CHECKING:
    from nova import Tensor


class MeanSquaredError(Metric):
    """
    Computes Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).

    MSE measures the average squared difference between predictions and targets:

    .. math::
        \\text{MSE} = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2

    When squared=False, computes RMSE (Root Mean Squared Error):

    .. math::
        \\text{RMSE} = \\sqrt{\\text{MSE}}

    MSE is commonly used for regression tasks as it penalizes larger errors
    more heavily than smaller ones. RMSE is in the same units as the target
    variable, making it more interpretable.

    Args:
        squared: If True, returns MSE. If False, returns RMSE. Default: True

    Examples:
        from nova.metrics import MSE
        >>>
        >>> # MSE for regression evaluation
        >>> mse = MSE(squared=True)
        >>> for batch in val_loader:
        ...     preds = model(batch['input'])
        ...     mse.update(preds, batch['target'])
        >>> print(f"MSE: {mse.compute():.4f}")

        >>> # RMSE (more interpretable)
        >>> rmse = MSE(squared=False)
        >>> rmse.update(nova.tensor([2.5, 0.0, 2.0]), nova.tensor([3.0, -0.5, 2.0]))
        >>> print(f"RMSE: {rmse.compute():.4f}")

        >>> # Track across multiple batches
        >>> mse = MSE()
        >>> for epoch in range(num_epochs):
        ...     mse.reset()
        ...     for batch in train_loader:
        ...         preds = model(batch['input'])
        ...         mse.update(preds, batch['target'])
        ...     print(f"Epoch {epoch} MSE: {mse.compute():.4f}")

    Note:
        - MSE is sensitive to outliers due to squaring
        - RMSE is in the same units as the target, making it easier to interpret
        - Both metrics assume regression (continuous outputs)
    """

    def __init__(self, squared: bool = True) -> None:
        """
        Initialize Mean Squared Error (MSE) or Root Mean Squared Error (RMSE) metric.

        Args:
            squared (bool, optional): If True, returns MSE. If False, returns RMSE
                (square root of MSE). Defaults to True.
        """
        super().__init__()
        self.squared = squared
        self.sum_squared_error = 0.0
        self.total = 0

    def reset(self) -> None:
        """Resets accumulated error and count to zero."""
        self.sum_squared_error = 0.0
        self.total = 0

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Accumulates squared errors from a batch.

        Args:
            preds: Predicted values.
            target: Ground truth values.

        Raises:
            ValueError: If preds and target shapes don't match.
        """
        self._check_dims(preds, target)
        diff = preds - target
        self.sum_squared_error += nova.sum((diff**2)).item()
        self.total += preds.numel()

    def compute(self) -> Tensor:
        """
        Computes the final MSE or RMSE.

        Returns:
            MSE if squared=True, RMSE if squared=False.
        """
        if self.total == 0:
            return nova.tensor(0.0)
        mse = self.sum_squared_error / self.total
        res = mse if self.squared else mse**0.5
        return nova.tensor(res)


class MeanAbsoluteError(Metric):
    """
    Computes Mean Absolute Error (MAE).

    MAE measures the average absolute difference between predictions and targets:

    .. math::
        \\text{MAE} = \\frac{1}{N} \\sum_{i=1}^{N} |y_i - \\hat{y}_i|

    MAE is more robust to outliers than MSE as it doesn't square the errors.
    It's also easier to interpret since it's in the same units as the target
    variable and represents the average magnitude of errors.

    Examples:
        >>> # MAE for house price prediction
        >>> mae = MAE()
        >>> predictions = nova.tensor([250000, 180000, 320000])  # Predicted prices
        >>> actual = nova.tensor([245000, 190000, 315000])  # Actual prices
        >>> mae.update(predictions, actual)
        >>> print(f"Average error: ${mae.compute():.2f}")

        >>> # Compare with MSE
        >>> mae = MAE()
        >>> mse = MSE()
        >>> preds = nova.tensor([1.0, 2.0, 10.0])  # Note the outlier
        >>> targets = nova.tensor([1.1, 2.1, 3.0])
        >>> mae.update(preds, targets)
        >>> mse.update(preds, targets)
        >>> print(f"MAE: {mae.compute():.2f}")  # Less affected by outlier
        >>> print(f"MSE: {mse.compute():.2f}")  # Heavily penalized by outlier

        >>> # Validation loop
        >>> mae = MAE()
        >>> model.eval()
        >>> for batch in val_loader:
        ...     with nova.no_grad():
        ...         preds = model(batch['input'])
        ...         mae.update(preds, batch['target'])
        >>> print(f"Validation MAE: {mae.compute():.4f}")

    Note:
        - More robust to outliers than MSE
        - All errors weighted equally (no squaring)
        - In same units as target variable
        - Preferred when outliers shouldn't dominate the metric
    """

    def __init__(self) -> None:
        """
        Initialize Mean Absolute Error (MAE) metric.

        Note:
            MAE is more robust to outliers than MSE as it doesn't square the errors.
        """
        super().__init__()
        self.sum_abs_error = 0.0
        self.total = 0

    def reset(self) -> None:
        """Resets accumulated absolute error and count to zero."""
        self.sum_abs_error = 0.0
        self.total = 0

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Accumulates absolute errors from a batch.

        Args:
            preds: Predicted values.
            target: Ground truth values.

        Raises:
            ValueError: If preds and target shapes don't match.
        """
        self._check_dims(preds, target)
        self.sum_abs_error += nova.sum(nova.abs((preds - target))).item()
        self.total += preds.numel()

    def compute(self) -> Tensor:
        """
        Computes the final MAE.

        Returns:
            Mean absolute error as a Tensor.
        """
        res = self.sum_abs_error / self.total if self.total > 0 else 0.0
        return nova.tensor(res)
