"""
Binary Classification Example - Stable Baseline
================================================

This example demonstrates binary classification with synthetic data designed
for training stability. Uses a simple, well-separated dataset to ensure
reliable convergence and proper metric computation.

Key concepts:
- Synthetic dataset with clear separation
- Stable training with proper initialization
- Binary cross-entropy with logits (numerical stability)
- Comprehensive metrics (accuracy, precision, recall, F1)
- Learning rate scheduling
- Model checkpointing
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import nova
import nova.nn as nn
import nova.nn.functional as F
from nova.optim import AdamW
from nova.optim.lr_scheduler import CosineAnnealingLR
from nova.metrics import Accuracy, Precision, Recall, F1Score
from nova.utils.data import Dataset, DataLoader, normalize
from nova.utils.logger import logger


class BinaryDataset(Dataset):
    """Simple dataset for binary classification."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_data(n_samples=2000, n_features=20, random_state=42):
    """Generate synthetic binary classification data with sklearn."""
    logger.info("Generating synthetic binary classification data...")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        flip_y=0.05,
        class_sep=1.5,
        random_state=random_state,
    )

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    logger.info(f"Generated {n_samples} samples with {n_features} features")
    logger.info(
        f"Class distribution - Class 0: {(y == 0).sum()}, Class 1: {(y == 1).sum()}"
    )

    return X, y


def load_data():
    """Create train/val/test datasets."""
    X, y = generate_data(n_samples=2000, n_features=20)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )

    X_train_t = nova.tensor(X_train, dtype=nova.float32)
    X_val_t = nova.tensor(X_val, dtype=nova.float32)
    X_test_t = nova.tensor(X_test, dtype=nova.float32)

    mean = X_train_t.mean(dim=0, keepdims=True)
    std = X_train_t.std(dim=0, keepdims=True)

    X_train_t = normalize(X_train_t, mean, std)
    X_val_t = normalize(X_val_t, mean, std)
    X_test_t = normalize(X_test_t, mean, std)

    y_train_t = nova.tensor(y_train, dtype=nova.float32).reshape(-1, 1)
    y_val_t = nova.tensor(y_val, dtype=nova.float32).reshape(-1, 1)
    y_test_t = nova.tensor(y_test, dtype=nova.float32).reshape(-1, 1)

    train_ds = BinaryDataset(X_train_t, y_train_t)
    val_ds = BinaryDataset(X_val_t, y_val_t)
    test_ds = BinaryDataset(X_test_t, y_test_t)

    logger.info(
        f"Split - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    return train_ds, val_ds, test_ds


class BinaryClassifier(nn.Module):
    """
    Simple feedforward network for binary classification.

    Architecture: 20 -> 64 -> 32 -> 1
    """

    def __init__(self, input_size=20):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


def train_epoch(model, loader, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        logits = model(X_batch)
        loss = F.binary_cross_entropy_with_logits(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, metrics):
    """Evaluate model and compute metrics."""
    model.eval()

    for metric in metrics.values():
        metric.reset()

    total_loss = 0.0

    with nova.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            probs = F.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch)

            for metric in metrics.values():
                metric.update(probs, y_batch)

            total_loss += loss.item()

    results = {name: metric.compute().item() for name, metric in metrics.items()}
    results["loss"] = total_loss / len(loader)

    return results


def main():
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-3

    print("=" * 70)
    logger.info("Binary Classification - Stable Synthetic Data")
    print("=" * 70)

    train_ds, val_ds, test_ds = load_data()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = BinaryClassifier(input_size=20)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    logger.info(f"\nModel:\n{model}")
    logger.info(f"Total parameters: {sum(p.data.size for p in model.parameters()):,}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: CosineAnnealingLR (T_max={EPOCHS})")

    train_metrics = {"accuracy": Accuracy(num_classes=2, task="binary")}

    val_metrics = {
        "accuracy": Accuracy(num_classes=2, task="binary"),
        "precision": Precision(num_classes=2, task="binary"),
        "recall": Recall(num_classes=2, task="binary"),
        "f1": F1Score(num_classes=2, task="binary"),
    }

    best_val_acc = 0.0

    print("\n" + "=" * 70)
    logger.info("Training started")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer)
        train_res = evaluate(model, train_loader, train_metrics)
        val_res = evaluate(model, val_loader, val_metrics)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch [{epoch:3d}/{EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} Acc: {train_res['accuracy']*100:5.2f}% | "
                f"Val Loss: {val_res['loss']:.4f} Acc: {val_res['accuracy']*100:5.2f}% "
                f"F1: {val_res['f1']*100:5.2f}% | LR: {current_lr:.6f}"
            )

        if val_res["accuracy"] > best_val_acc:
            best_val_acc = val_res["accuracy"]
            nova.save(model.state_dict(), "best_binary_model.pth")
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"  → Best model saved | "
                    f"Precision: {val_res['precision']*100:.2f}% "
                    f"Recall: {val_res['recall']*100:.2f}%"
                )

    print("\n" + "=" * 70)
    logger.info("Evaluating on test set...")
    print("=" * 70)

    model.load_state_dict(nova.load("best_binary_model.pth"))

    test_metrics = {
        "accuracy": Accuracy(num_classes=2, task="binary"),
        "precision": Precision(num_classes=2, task="binary"),
        "recall": Recall(num_classes=2, task="binary"),
        "f1": F1Score(num_classes=2, task="binary"),
    }

    test_res = evaluate(model, test_loader, test_metrics)

    logger.info(f"Test Loss: {test_res['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_res['accuracy']*100:.2f}%")
    logger.info(f"Test Precision: {test_res['precision']*100:.2f}%")
    logger.info(f"Test Recall: {test_res['recall']*100:.2f}%")
    logger.info(f"Test F1 Score: {test_res['f1']*100:.2f}%")

    print("=" * 70)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
