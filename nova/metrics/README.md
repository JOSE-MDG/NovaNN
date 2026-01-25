# `metrics` Module

The **`metrics/`** directory implements **model evaluation metrics** for classification and regression tasks.

Unlike loss functions, metrics **are not used for backpropagation** and are designed to **accumulate statistics across multiple batches**, providing a comprehensive evaluation of model performance during training and validation.

All metrics follow a consistent three-step pattern: `reset()` → `update()` → `compute()`.

## General Structure

The `metrics/` module is organized into:

- **`metric.py`**: Abstract base class `Metric` that defines the common interface
- **[`classification/`](#submodule-classification)**: Metrics for classification tasks
- **[`regression/`](#submodule-regression)**: Metrics for regression tasks

## Base Class `Metric`

Defines the abstract interface that all metrics must implement.

**Required methods:**

- **`reset()`**: Clears accumulated internal state
  - Called at the beginning of each epoch to start from scratch
  - Initializes counters, sums, and statistics to their default values

- **`update(preds, targets)`**: Accumulates statistics from a batch
  - Receives predictions and targets of the current batch
  - Updates internal counters without computing the final value
  - Called once per batch during training/evaluation

- **`compute()`**: Calculates and returns the final metric value
  - Uses statistics accumulated from all `update()` calls
  - Called at the end of the epoch to obtain the result
  - Does not reset the state (must call `reset()` manually)

**Utilities:**

- **`_check_dims(preds, targets)`**: Validates that shapes match

**Typical usage pattern:**

```python
metric = SomeMetric()

for epoch in range(num_epochs):
    metric.reset()  # Clear for new epoch

    for x, y in loader:
        out = model(x)
        metric.update(out, y)  # Accumulate

    score = metric.compute().item()  # Calculate final result
    print(f"Epoch {epoch}: {score}")
```

## Submodule `classification/`

Contains metrics for evaluating classification models.

### `_confusion.py`

**`ConfusionMatrix(num_classes)`**: Multi-class confusion matrix.

**Features:**

- **Structure**: Matrix (num_classes, num_classes)
  - Row i = true class
  - Column j = predicted class
  - Diagonal = correct predictions (TP)
  - Off-diagonal = errors
- **Efficient implementation**: Uses `np.bincount` for fast counting
- **Auto-argmax**: If it receives probabilities/logits (N, C), automatically applies argmax
- **Validation**: Filters invalid indices to avoid errors

**Formula:**

```
C[i,j] = count(y_true == i and y_pred == j)
```

### `_stat.py`

Contains metrics derived from the confusion matrix.

**Base class `ClassificationStat(num_classes, average)`:**

All classification metrics inherit from this class, which:

- Maintains an internal `ConfusionMatrix`
- Automatically calculates TP, FP, TN, FN
- Supports different averaging strategies:
  - **`'micro'`**: Calculates metrics globally (sums all TP, FP, FN)
  - **`'macro'`**: Calculates per class and averages without weights
  - **`'weighted'`**: Calculates per class and averages weighted by support
  - **`None`**: Returns score for each class (array of size num_classes)

#### `Accuracy(num_classes, average='micro')`

Proportion of correct predictions.

**Formula:**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretation:**

- "What percentage of all predictions were correct?"
- Value between 0.0 and 1.0
- Can be misleading in imbalanced datasets

#### `Precision(num_classes, average='macro')`

Proportion of positive predictions that were correct.

**Formula:**

```
Precision = TP / (TP + FP)
```

**Interpretation:**

- "Of all samples I predicted as Positive, how many actually were?"
- High precision = few false positives
- Important when the cost of FP is high

#### `Recall(num_classes, average='macro')`

Proportion of actual positives that were found.

**Formula:**

```
Recall = TP / (TP + FN)
```

**Interpretation:**

- "Of all samples that were actually Positive, how many did I find?"
- High recall = few false negatives
- Important when the cost of FN is high (e.g., disease detection)

#### `F1Score(num_classes, average='macro')`

Harmonic mean of Precision and Recall.

**Formula:**

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Interpretation:**

- Balance between Precision and Recall
- Useful when a single number summarizing both metrics is needed
- Value between 0.0 and 1.0
- High F1 requires both high precision and high recall

### `_roc_auc.py`

**`ROCAUC(num_classes=2)`**: Area under the ROC curve.

**Features:**

- **Stores all data**: Unlike other metrics, stores all predictions and targets in memory
- **ROC curve**: Graph of TPR (True Positive Rate) vs FPR (False Positive Rate) for different thresholds
- **AUC**: Area under the ROC curve (integral using trapezoidal rule)
- **Automatic detach**: Releases gradients to avoid memory leaks
- **Mainly binary**: Optimized for binary classification

**Interpretation:**

- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier (no better than flipping a coin)
- AUC < 0.5: Classifier worse than random

**Warning:**

- Consumes a lot of memory on large datasets (stores everything)
- Not recommended for extremely large datasets

## Submodule `regression/`

Contains metrics for evaluating regression models.

### `_error.py`

#### `MeanSquaredError(squared=True)`

Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).

**Formulas:**

```
MSE = (1/N) * Σ(y_true - y_pred)²
RMSE = √MSE
```

**Features:**

- **MSE** (`squared=True`): Heavily penalizes large errors (due to squaring)
- **RMSE** (`squared=False`): In the same units as the target (more interpretable)
- Sensitive to outliers
- Differentiable (also used as loss)

**When to use:**

- MSE: When large errors must be heavily penalized
- RMSE: When interpretability in original units is needed

#### `MeanAbsoluteError()`

Mean Absolute Error (MAE).

**Formula:**

```
MAE = (1/N) * Σ|y_true - y_pred|
```

**Features:**

- More robust to outliers than MSE
- All errors weighted equally
- In the same units as the target
- Easy to interpret

**When to use:**

- When outliers should not dominate the metric
- When uniform treatment of errors is preferred

### `_r2.py`

**`R2Score()`**: Coefficient of determination (R²).

**Formula:**

```
R² = 1 - (SS_res / SS_tot)
   = 1 - (Σ(y - ŷ)² / Σ(y - ȳ)²)
```

**Interpretation:**

- **R² = 1.0**: Perfect model (explains all variance)
- **R² = 0.0**: Model as good as predicting the mean
- **R² < 0.0**: Model worse than predicting the mean

**Features:**

- Measures what proportion of variance is explained by the model
- Normalized (doesn't depend on units)
- Can be negative if the model is very bad
- Sensitive to outliers (like MSE)

## Design and Philosophy

The `metrics` module of NovaNN is designed following these principles:

- **Consistent pattern**: All metrics follow reset() → update() → compute()
- **Efficient accumulation**: Maintain only necessary statistics, not all data (except ROCAUC)
- **Separation of concerns**: Metrics don't compute gradients, only evaluate
- **Flexibility**: Support averaging strategies for multi-class
- **Tensor integration**: Work directly with NovaNN Tensor objects

## Integration with other modules

The `metrics` module integrates with:

- **[`nn/`](../nn/README.md)**: Evaluates model outputs during training/validation
- **[`autograd/`](../autograd/README.md)**: Metrics do `.detach()` internally to avoid consuming graph memory
- **Tensors**: All metrics operate on `Tensor` objects

## Usage Examples

### Example 1: Accuracy in binary classification

```python
from nova.metrics import Accuracy

acc = Accuracy(num_classes=2, average='micro')
preds = nova.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])  # Probabilities
targets = nova.tensor([0, 1, 0])  # True classes
acc.update(preds, targets)
print(f"Accuracy: {acc.compute():.2%}")  # 100%
```

### Example 2: Precision, Recall, F1 in multi-class

```python
from nova.metrics import Precision, Recall, F1Score

precision = Precision(num_classes=3, average='macro')
recall = Recall(num_classes=3, average='macro')
f1 = F1Score(num_classes=3, average='macro')

# Simulate predictions for a batch
logits = nova.randn(16, 3)  # 16 samples, 3 classes
targets = nova.randint(0, 3, (16,))

precision.update(logits, targets)
recall.update(logits, targets)
f1.update(logits, targets)

print(f"Precision: {precision.compute():.4f}")
print(f"Recall: {recall.compute():.4f}")
print(f"F1: {f1.compute():.4f}")
```

### Example 3: ROC-AUC for binary classification

```python
from nova.metrics import ROCAUC

auc = ROCAUC(num_classes=2)

# Probabilities for positive class
probs = nova.tensor([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]])
targets = nova.tensor([1, 1, 1, 0])
auc.update(probs, targets)
print(f"AUC: {auc.compute():.4f}")
```

### Example 4: MSE and MAE for regression

```python
from nova.metrics import MSE, MAE

mse = MSE(squared=True)
rmse = MSE(squared=False)
mae = MAE()

predictions = nova.tensor([2.5, 0.0, 2.0, 8.0])
targets = nova.tensor([3.0, -0.5, 2.0, 7.0])

mse.update(predictions, targets)
rmse.update(predictions, targets)
mae.update(predictions, targets)

print(f"MSE: {mse.compute():.4f}")
print(f"RMSE: {rmse.compute():.4f}")
print(f"MAE: {mae.compute():.4f}")
```

### Example 5: R² Score

(python example 6)

```python
from nova.metrics import R2Score

r2 = R2Score()
preds = nova.tensor([3.0, 2.5, 4.0, 5.5])
targets = nova.tensor([3.2, 2.4, 4.1, 5.0])
r2.update(preds, targets)
print(f"R² Score: {r2.compute():.4f}") # Close to 1.0 = good fit
```

### Example 6: Complete training loop with multiple metrics

```python
import nova.nn as nn
from nova.metrics import Accuracy, F1Score, Precision, Recall

model = nn.Sequential(nn.Linear(784, 10))
criterion = nn.CrossEntropyLoss()

# Metrics
train_acc = Accuracy(num_classes=10)
val_acc = Accuracy(num_classes=10)
val_f1 = F1Score(num_classes=10, average='weighted')

for epoch in range(10): # Training
    model.train()
    train_acc.reset()
    for batch in train_loader:
        preds = model(batch['input'])
        loss = criterion(preds, batch['target']) # ... backward ...
        train_acc.update(preds, batch['target'])

    # Validation
    model.eval()
    val_acc.reset()
    val_f1.reset()
    with nova.no_grad():
        for batch in val_loader:
            preds = model(batch['input'])
            val_acc.update(preds, batch['target'])
            val_f1.update(preds, batch['target'])

    print(f"Epoch {epoch}:")
    print(f"  Train Acc: {train_acc.compute():.2%}")
    print(f"  Val Acc: {val_acc.compute():.2%}")
    print(f"  Val F1: {val_f1.compute():.4f}")
```

### Example 7: Confusion Matrix

```python
from nova.metrics import ConfusionMatrix

cm = ConfusionMatrix(num_classes=3)
preds = nova.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.3, 0.5]])
targets = nova.tensor([1, 0, 2])
cm.update(preds, targets)
matrix = cm.compute()
print("Confusion Matrix:")
print(matrix)
```

---

> For more details on specific implementations, consult the source code in `classification/` and `regression/`.
