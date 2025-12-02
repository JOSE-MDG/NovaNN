import numpy as np
from typing import Callable, Any
from novann._typing import Loader
from novann.model import Sequential


def accuracy(
    model: Sequential,
    data_loader: Loader,
) -> float:
    """Compute classification accuracy for a model over a dataloader.

    Args:
        model: Callable that maps a batch of inputs X (np.ndarray) to logits/probabilities.
        data_loader: Iterator that yields (X_batch, y_batch) tuples. y_batch should
            contain integer class labels.

    Returns:
        Fraction of correctly predicted samples (float in [0, 1]).
    """
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in data_loader:
        y_pred = model(X_batch)
        pred_classes = np.argmax(y_pred, axis=1)
        total_correct += np.sum(pred_classes == y_batch)
        total_samples += y_batch.shape[0]

    return total_correct / total_samples


def binary_accuracy(
    model: Sequential,
    data_loader: Loader,
) -> float:
    """Compute binary classification accuracy for a model over a dataloader.

    Args:
        model: Callable that maps a batch of inputs X (np.ndarray) to probabilities [0, 1].
        data_loader: Iterator that return (X_batch, y_batch) tuples. y_batch should
            contain binary class labels (0 or 1).

    Returns:
        Fraction of correctly predicted samples (float in [0, 1]).
    """
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in data_loader:
        # Get the model's output
        y_pred = model(X_batch)

        # Convert probabilities to binary prediction (0 or 1) using a 0.5 threshold.
        pred_classes = (y_pred >= 0.5).astype(np.int64)

        # Accumulate correct predictions and total samples across batches
        total_correct += np.sum(pred_classes == y_batch)
        total_samples += y_batch.shape[0]

    return total_correct / total_samples


def r2_score(
    model: Sequential,
    data_loader: Loader,
) -> float:
    """Calculate the R² score (Coefficient of Determination) for a regression model.

    Args:
        model: Callable that maps a batch of inputs X (np.ndarray) to predicted values.
        data_loader: Iterator that return (X_batch, y_batch) tuples.

    Returns:
        The R² value (float).
    """
    all_y_true = []
    all_y_pred = []

    for X_batch, y_batch in data_loader:
        # Get the model's prediction for the current batch
        y_pred = model(X_batch)

        # Flatten and accumulate predictions and true values
        all_y_pred.append(y_pred.flatten())
        all_y_true.append(y_batch.flatten())

    # Concatenate all accumulated values from all batches
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    # --- FINAL R^2 CALCULATION ---

    # Calculate Sum of Squares of Residuals (SSE)
    # This is the unexplained variance.
    sse = np.sum((y_true - y_pred) ** 2)

    # Calculate Total Sum of Squares (SST)
    # This is the total variance in the data.
    sst = np.sum((y_true - np.mean(y_true)) ** 2)

    # Handle the case where there is no variance in true values
    if sst == 0:
        return 1.0  # Perfect fit (though data should likely be checked)

    # R^2 = 1 - (SSE / SST)
    return 1 - (sse / sst)
