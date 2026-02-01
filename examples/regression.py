"""
Regression Example - Stable Baseline
=====================================

This example demonstrates regression with synthetic data designed for
training stability. Uses a simple polynomial relationship to ensure
reliable convergence and proper metric computation.

Key concepts:
- Synthetic dataset with known relationship
- MSE loss for regression
- Regression metrics (MSE, MAE, R2 Score)
- Stable training with proper initialization
- Feature normalization
- Model checkpointing
"""

import numpy as np
from sklearn.model_selection import train_test_split

import nova
import nova.nn as nn
import nova.nn.functional as F
from nova.optim import AdamW
from nova.nn.utils import clip_grad_norm_
from nova.metrics import MSE, MAE, R2Score
from nova.utils.data import Dataset, DataLoader, normalize
from nova.utils.logger import get_logger

logger = get_logger()

# Dataset


class RegressionDataset(Dataset):
    """Simple dataset for regression."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# Data Generation


def generate_data(n_samples=5000, n_features=10, noise=0.1, random_state=42):
    """
    Generate synthetic regression data with polynomial relationship.

    y = 3*x1 + 2*x2 - 1.5*x3 + 0.5*x1*x2 + noise

    This creates a learnable pattern with controlled complexity.
    """
    logger.info("Generating synthetic regression data...")

    np.random.seed(random_state)

    # Generate features from normal distribution
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create target with known relationship
    y = (
        3.0 * X[:, 0]  # Linear term
        + 2.0 * X[:, 1]  # Linear term
        + -1.5 * X[:, 2]  # Linear term
        + 0.5 * X[:, 0] * X[:, 1]  # Interaction term
    )

    # Add Gaussian noise
    y += np.random.randn(n_samples) * noise
    y = y.astype(np.float32)

    logger.info(f"Generated {n_samples} samples with {n_features} features")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Target statistics - Mean: {y.mean():.3f}, Std: {y.std():.3f}")

    return X, y


def load_data():
    """Create train/val/test datasets."""
    X, y = generate_data(n_samples=5000, n_features=10, noise=0.1)

    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Convert to tensors
    X_train_t = nova.tensor(X_train, dtype=nova.float32)
    X_val_t = nova.tensor(X_val, dtype=nova.float32)
    X_test_t = nova.tensor(X_test, dtype=nova.float32)

    # Standardize features (using training statistics)
    x_mean = X_train_t.mean(dim=0, keepdims=True)
    x_std = X_train_t.std(dim=0, keepdims=True)

    X_train_t = normalize(X_train_t, x_mean, x_std)
    X_val_t = normalize(X_val_t, x_mean, x_std)
    X_test_t = normalize(X_test_t, x_mean, x_std)

    # Labels to (N, 1)
    y_train_t = nova.tensor(y_train, dtype=nova.float32).reshape(-1, 1)
    y_val_t = nova.tensor(y_val, dtype=nova.float32).reshape(-1, 1)
    y_test_t = nova.tensor(y_test, dtype=nova.float32).reshape(-1, 1)

    # Standardize labels (using training statistics)
    y_mean = y_train_t.mean()
    y_std = y_train_t.std()

    y_train_t = normalize(y_train_t, y_mean, y_std)
    y_val_t = normalize(y_val_t, y_mean, y_std)
    y_test_t = normalize(y_test_t, y_mean, y_std)

    # Create datasets
    train_ds = RegressionDataset(X_train_t, y_train_t)
    val_ds = RegressionDataset(X_val_t, y_val_t)
    test_ds = RegressionDataset(X_test_t, y_test_t)

    logger.info(
        f"Split - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    return train_ds, val_ds, test_ds


# Model


class RegressionMLP(nn.Module):
    """
    Simple feedforward network for regression.

    Architecture: 10 -> 32 -> 16 -> 1
    Uses ReLU activation and He initialization for stability.
    """

    def __init__(self, input_size=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self._initialize_weights()

    # He initialization for ReLU
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# Training & Evaluation


def train_epoch(model, loader, optimizer, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        pred = model(X_batch)
        loss = F.mse_loss(pred, y_batch)

        loss.backward()
        norms = clip_grad_norm_(model.parameters(), max_norm=1.0, get_norm=True)
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader), norms


def evaluate(model, loader, metrics):
    """Evaluate model and compute metrics."""
    model.eval()

    for metric in metrics.values():
        metric.reset()

    total_loss = 0.0

    with nova.no_grad():
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            loss = F.mse_loss(pred, y_batch)

            for metric in metrics.values():
                metric.update(pred, y_batch)

            total_loss += loss.item()

    results = {name: metric.compute().item() for name, metric in metrics.items()}
    results["loss"] = total_loss / len(loader)

    return results


# Main


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-4

    print("=" * 70)
    logger.info("Regression - Stable Synthetic Data")
    print("=" * 70)

    # Load data
    train_ds, val_ds, test_ds = load_data()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # Model and optimizer
    model = RegressionMLP(input_size=10)
    optimizer = AdamW(
        model.parameters(), lr=LR, weight_decay=1e-5
    )  # Increased weight decay

    logger.info(f"\nModel:\n{model}")
    logger.info(f"Total parameters: {sum(p.data.size for p in model.parameters()):,}")
    logger.info(f"Optimizer: {optimizer}")

    # Metrics
    train_metrics = {"mse": MSE()}
    val_metrics = {"mse": MSE(), "mae": MAE(), "r2": R2Score()}

    best_val_loss = float("inf")

    print("\n" + "=" * 70)
    logger.info("Training started")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, norms = train_epoch(model, train_loader, optimizer)
        train_res = evaluate(model, train_loader, train_metrics)

        # Validate
        val_res = evaluate(model, val_loader, val_metrics)

        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == 1:

            logger.info(
                f"Epoch [{epoch:3d}/{EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} RMSE: {train_res['mse']**0.5:.4f} | "
                f"Val Loss: {val_res['loss']:.4f} RMSE: {val_res['mse']**0.5:.4f} "
                f"MAE: {val_res['mae']:.4f}"
            )
            logger.info(f"R2: {val_res['r2']:.4f} | Gradient norms {norms}\n")

        # Save best model
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            nova.save(model.state_dict(), "best_regression.pth")
            if epoch % 10 == 0 or epoch == 1:
                logger.info("  → Best model saved")

    # Test evaluation
    print("\n" + "=" * 70)
    logger.info("Evaluating on test set...")
    print("=" * 70)

    model.load_state_dict(nova.load("best_regression.pth"))

    test_metrics = {"mse": MSE(), "mae": MAE(), "r2": R2Score()}
    test_res = evaluate(model, test_loader, test_metrics)

    logger.info(f"Test MSE: {test_res['mse']:.4f}")
    logger.info(f"Test RMSE: {test_res['mse']**0.5:.4f}")
    logger.info(f"Test MAE: {test_res['mae']:.4f}")
    logger.info(f"Test R2 Score: {test_res['r2']:.4f}")

    print("=" * 70)
    logger.info("Training complete!")

    # Show expected performance
    logger.info("\nExpected performance:")
    logger.info("  - R2 Score should be > 0.95 (excellent fit)")
    logger.info("  - RMSE should be < 0.15 (low error)")
    logger.info("  - Loss should decrease monotonically")


if __name__ == "__main__":
    main()
