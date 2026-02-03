from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from nova import Tensor


class Metric(ABC):
    """
    Abstract base class for all metrics in NovaNN.

    Metrics are used to evaluate model performance during training and validation.
    Unlike loss functions, metrics are not used for backpropagation and can track
    statistics across multiple batches.

    The typical workflow for using a metric is:
    1. Initialize the metric
    2. Call update() for each batch with predictions and targets
    3. Call compute() to get the final metric value
    4. Call reset() to clear accumulated statistics for the next epoch

    Subclasses must implement:
    - reset(): Clear internal state
    - update(): Accumulate statistics from a batch
    - compute(): Calculate final metric value

    Examples:
        >>> # Custom metric example
        >>> class MyMetric(Metric):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.correct = 0
        ...         self.total = 0
        ...
        ...     def reset(self):
        ...         self.correct = 0
        ...         self.total = 0
        ...
        ...     def update(self, preds, targets):
        ...         self._check_dims(preds, targets)
        ...         self.correct += (preds == targets).sum().item()
        ...         self.total += targets.numel()
        ...
        ...     def compute(self):
        ...         return self.correct / self.total if self.total > 0 else 0.0

        >>> # Usage in training loop
        >>> metric = MyMetric()
        >>> for epoch in range(num_epochs):
        ...     metric.reset()  # Clear for new epoch
        ...     for batch in dataloader:
        ...         preds = model(batch['input'])
        ...         metric.update(preds, batch['target'])
        ...     print(f"Epoch {epoch}: {metric.compute()}")

    Note:
        Metrics accumulate statistics in memory and should be reset between
        epochs to avoid incorrect calculations.
    """

    def __init__(self):
        """
        Initialize the metric and reset internal state.

        Note:
            Automatically calls reset() to initialize accumulation variables.
            Subclasses should implement reset(), update(), and compute() methods.
        """
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the metric's internal state.

        This method should clear all accumulated statistics, preparing the
        metric for a new epoch or evaluation phase. Called automatically
        during initialization.

        Examples:
            >>> metric = Accuracy()
            >>> metric.update(preds, targets)
            >>> metric.reset()  # Clear accumulated stats
            >>> metric.update(new_preds, new_targets)  # Fresh start
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, preds: Tensor, targets: Tensor) -> None:
        """
        Updates the metric's state with a new batch of predictions.

        This method accumulates statistics from the current batch without
        computing the final metric value. Designed to be called once per
        batch during training or evaluation.

        Args:
            preds (Tensor): Model predictions for the current batch.
            targets (Tensor): Ground truth labels/values for the current batch.

        Raises:
            ValueError: If preds and targets have mismatched shapes.

        Examples:
            >>> metric = MSE()
            >>> for batch in dataloader:
            ...     preds = model(batch['input'])
            ...     metric.update(preds, batch['target'])

        Note:
            This method should not return a value. Use compute() to get
            the final metric result after all batches are processed.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        """
        Computes and returns the final metric value.

        This method calculates the metric using all accumulated statistics
        from previous update() calls. Should be called after processing all
        batches in an epoch or evaluation phase.

        Returns:
            Final metric value as a float or Tensor.

        Examples:
            >>> metric = Accuracy()
            >>> for batch in dataloader:
            ...     metric.update(model(batch['input']), batch['target'])
            >>> final_accuracy = metric.compute()
            >>> print(f"Accuracy: {final_accuracy:.2%}")

        Note:
            This method does not reset the metric. Call reset() manually
            before starting a new epoch.
        """
        raise NotImplementedError

    def _check_dims(self, preds: Tensor, targets: Tensor):
        """
        Validates that predictions and targets have matching shapes.

        Args:
            preds (Tensor): Predicted values.
            targets (Tensor): Ground truth values.

        Raises:
            ValueError: If shapes don't match.

        Note:
            This is a utility method for subclasses to validate inputs
            in their update() methods.
        """
        if preds.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: preds {preds.shape} != targets {targets.shape}"
            )
