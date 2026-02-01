"""
Multiclass Classification Example
==================================

This example demonstrates how to build and train a feedforward neural network
for multiclass classification using NovaNN with the sklearn digits dataset.

Key concepts:
- Custom Dataset implementation
- DataLoader for batching and shuffling
- Fully connected layers (Linear)
- Dropout for regularization
- Cross-entropy loss for multiclass problems
- Learning rate scheduling with StepLR
- Comprehensive metrics (accuracy, precision, recall, F1)
- Confusion matrix analysis
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import nova
import nova.nn as nn
import nova.nn.functional as F
from nova.optim import AdamW
from nova.optim.lr_scheduler import StepLR
from nova.metrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from nova.utils.data import Dataset, DataLoader, normalize
from nova.utils.logger import get_logger

logger = get_logger()


# Custom Dataset


class DigitsDataset(Dataset):
    """Custom dataset for sklearn digits."""

    def __init__(self, features, labels):
        """
        Args:
            features: Tensor of shape (N, 64)
            labels: Tensor of shape (N,)
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# Data Loading & Preprocessing


def load_and_preprocess_data():
    """
    Load sklearn digits dataset and create NovaNN datasets.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Loading sklearn digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target

    n_classes = len(nova.unique(nova.tensor(y)))
    logger.info(
        f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes"
    )

    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )

    # Convert to tensors and normalize
    X_train_t = nova.tensor(X_train, dtype=nova.float32)
    X_val_t = nova.tensor(X_val, dtype=nova.float32)
    X_test_t = nova.tensor(X_test, dtype=nova.float32)

    # Standardize using training statistics
    mean = X_train_t.mean(dim=0, keepdims=True)
    std = X_train_t.std(dim=0, keepdims=True)

    X_train_t = normalize(X_train_t, mean, std)
    X_val_t = normalize(X_val_t, mean, std)
    X_test_t = normalize(X_test_t, mean, std)

    # Convert labels to tensors
    y_train_t = nova.tensor(y_train, dtype=nova.long)
    y_val_t = nova.tensor(y_val, dtype=nova.long)
    y_test_t = nova.tensor(y_test, dtype=nova.long)

    # Create datasets
    train_dataset = DigitsDataset(X_train_t, y_train_t)
    val_dataset = DigitsDataset(X_val_t, y_val_t)
    test_dataset = DigitsDataset(X_test_t, y_test_t)

    logger.info(
        f"Split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


# Model Definition


class DigitsClassifier(nn.Module):
    """
    Feedforward neural network for digit classification.

    Architecture:
        Linear(64, 128) -> ReLU -> Dropout(0.3)
        Linear(128, 64) -> ReLU -> Dropout(0.3)
        Linear(64, 32) -> ReLU -> Dropout(0.2)
        Linear(32, 10)
    """

    def __init__(self, input_size: int = 64, num_classes: int = 10):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(32, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x


# Training & Evaluation


def train_epoch(model, loader, optimizer):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, metrics):
    """Evaluate model and update metrics."""
    model.eval()

    # Reset all metrics
    for metric in metrics.values():
        metric.reset()

    total_loss = 0.0

    with nova.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)

            # Update all metrics
            for metric in metrics.values():
                metric.update(logits, y_batch)

            total_loss += loss.item()

    # Compute final metrics
    results = {}
    for name, metric in metrics.items():
        results[name] = (
            metric.compute()
            if isinstance(metric, ConfusionMatrix)
            else metric.compute().item()
        )
        results["loss"] = total_loss / len(loader)

    return results


# Main Training Pipeline


def main():
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    STEP_SIZE = 15
    GAMMA = 0.5

    print("=" * 70)
    logger.info("Multiclass Classification - Digits Dataset (sklearn)")
    print("=" * 70)

    # Load and preprocess data
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model, optimizer, scheduler
    model = DigitsClassifier(input_size=64, num_classes=10)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    logger.info(f"\nModel:\n{model}")
    logger.info(f"\nTotal parameters: {sum(p.data.size for p in model.parameters()):,}")
    logger.info(f"Optimizer:\n{optimizer}")
    logger.info(f"Scheduler: StepLR (step_size={STEP_SIZE}, gamma={GAMMA})")

    # Initialize metrics
    train_metrics = {"accuracy": Accuracy(num_classes=10)}

    val_metrics = {
        "accuracy": Accuracy(num_classes=10),
        "precision": Precision(num_classes=10, average="macro"),
        "recall": Recall(num_classes=10, average="macro"),
        "f1": F1Score(num_classes=10, average="macro"),
    }

    # Training loop
    best_val_acc = 0.0
    print("\n" + "=" * 70)
    logger.info("Training started")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer)
        train_results = evaluate(model, train_loader, train_metrics)

        # Validate
        val_results = evaluate(model, val_loader, val_metrics)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging
        logger.info(
            f"Epoch [{epoch:2d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_results['accuracy']*100:5.2f}% | "
            f"Val Loss: {val_results['loss']:.4f} Acc: {val_results['accuracy']*100:5.2f}% "
            f"F1: {val_results['f1']*100:5.2f}% | LR: {current_lr:.6f}"
        )

        # Save best model
        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            nova.save(model.state_dict(), "best_digits_model.pth")
            logger.info(
                f"  → Best model saved | "
                f"Precision: {val_results['precision']*100:.2f}% "
                f"Recall: {val_results['recall']*100:.2f}%"
            )

    # Final evaluation on test set
    print("\n" + "=" * 70)
    logger.info("Evaluating on test set...")
    print("=" * 70)

    model.load_state_dict(nova.load("best_digits_model.pth"))

    test_metrics = {
        "accuracy": Accuracy(num_classes=10),
        "precision": Precision(num_classes=10, average="macro"),
        "recall": Recall(num_classes=10, average="macro"),
        "f1": F1Score(num_classes=10, average="macro"),
        "confusion": ConfusionMatrix(num_classes=10),
    }

    test_results = evaluate(model, test_loader, test_metrics)

    logger.info(f"Test Loss: {test_results['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_results['accuracy']*100:.2f}%")
    logger.info(f"Test Precision: {test_results['precision']*100:.2f}%")
    logger.info(f"Test Recall: {test_results['recall']*100:.2f}%")
    logger.info(f"Test F1 Score: {test_results['f1']*100:.2f}%")

    # Display confusion matrix
    cm = test_results["confusion"]
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")

    print("=" * 70)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
