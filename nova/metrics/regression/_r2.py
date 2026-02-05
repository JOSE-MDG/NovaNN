from __future__ import annotations
import nova
from typing import TYPE_CHECKING
from nova.metrics.metric import Metric

if TYPE_CHECKING:
    from nova import Tensor


class R2Score(Metric):
    """
    Computes the R² (coefficient of determination) score.

    R² represents the proportion of variance in the target variable that is
    predictable from the input features. It provides a measure of how well
    the model explains the variability of the data:

    .. math::
        R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum_i (y_i - \\hat{y}_i)^2}{\\sum_i (y_i - \\bar{y})^2}

    where:
    - SS_res is the residual sum of squares (model errors)
    - SS_tot is the total sum of squares (variance in data)
    - ȳ is the mean of target values

    **Interpretation:**
    - R² = 1.0: Perfect predictions (all variance explained)
    - R² = 0.0: Model performs as well as predicting the mean
    - R² < 0.0: Model performs worse than predicting the mean

    Examples:
        >>> # Evaluate regression model
        >>> r2 = R2Score()
        >>> predictions = nova.tensor([3.0, 2.5, 4.0, 5.5])
        >>> targets = nova.tensor([3.2, 2.4, 4.1, 5.0])
        >>> r2.update(predictions, targets)
        >>> print(f"R² Score: {r2.compute():.4f}")  # Close to 1.0 = good fit

        >>> # Compare models
        >>> r2_model1 = R2Score()
        >>> r2_model2 = R2Score()
        >>> for batch in test_loader:
        ...     r2_model1.update(model1(batch['input']), batch['target'])
        ...     r2_model2.update(model2(batch['input']), batch['target'])
        >>> print(f"Model 1 R²: {r2_model1.compute():.4f}")
        >>> print(f"Model 2 R²: {r2_model2.compute():.4f}")

        >>> # Training loop with R² tracking
        >>> for epoch in range(num_epochs):
        ...     model.train()
        ...     # ... training code ...
        ...
        ...     model.eval()
        ...     r2 = R2Score()
        ...     for batch in val_loader:
        ...         with nova.no_grad():
        ...             preds = model(batch['input'])
        ...             r2.update(preds, batch['target'])
        ...     print(f"Epoch {epoch} R²: {r2.compute():.4f}")

        >>> # Detect poor model
        >>> r2 = R2Score()
        >>> preds = nova.randn(100)  # Random predictions
        >>> targets = nova.randn(100)
        >>> r2.update(preds, targets)
        >>> score = r2.compute()
        >>> if score < 0.5:
        ...     print("Warning: Model explains less than 50% of variance!")

    Note:
        - R² ranges from -∞ to 1.0 (though typically 0.0 to 1.0)
        - Values closer to 1.0 indicate better fit
        - Can be negative if model is worse than baseline
        - Sensitive to outliers (like MSE)
        - Best used alongside other metrics (MSE, MAE)
        - Not suitable for comparing models trained on different datasets

    Reference:
        - Coefficient of determination: https://en.wikipedia.org/wiki/Coefficient_of_determination
        - Scikit-learn R² score: https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
    """

    def __init__(self) -> None:
        """
        Initialize R² (coefficient of determination) score metric.

        Note:
            R² measures the proportion of variance in the target that is predictable
            from the model. Values range from -∞ to 1.0, where 1.0 indicates perfect
            predictions and values below 0.0 indicate the model performs worse than
            predicting the mean.
        """
        super().__init__()
        self.sum_squared_error = 0.0
        self.sum_target = 0.0
        self.sum_target_sq = 0.0
        self.total = 0

    def reset(self) -> None:
        """Resets all accumulated statistics to zero."""
        self.sum_squared_error = 0.0
        self.sum_target = 0.0
        self.sum_target_sq = 0.0
        self.total = 0

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Accumulates statistics needed for R² computation.

        Args:
            preds: Predicted values.
            target: Ground truth values.

        Raises:
            ValueError: If preds and target shapes don't match.

        Note:
            Accumulates three statistics:
            - Sum of squared errors (residuals)
            - Sum of targets (for mean calculation)
            - Sum of squared targets (for total variance)
        """
        self._check_dims(preds, target)
        self.sum_squared_error += nova.sum(((target - preds) ** 2)).item()
        self.sum_target += nova.sum(target).item()
        self.sum_target_sq += nova.sum((target**2)).item()
        self.total += target.numel()

    def compute(self) -> Tensor:
        """
        Computes the final R² score.

        Returns:
            R² score as a Tensor. Returns 0.0 if no samples or if
            total variance is zero (all targets are identical).

        Note:
            - Returns 0.0 for degenerate cases (zero variance in targets)
            - Can return negative values if model is very poor
        """
        if self.total == 0:
            return nova.tensor(0.0)

        # Total sum of squares: SS_tot = Σ(y - ȳ)²
        ss_tot = self.sum_target_sq - (self.sum_target**2 / self.total)

        # Avoid division by zero (all targets are identical)
        if ss_tot == 0:
            return nova.tensor(0.0)

        # R² = 1 - (SS_res / SS_tot)
        res = 1 - (self.sum_squared_error / ss_tot)
        return nova.tensor(res)
