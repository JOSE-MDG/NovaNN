"""
Convolutional Neural Network Example
=====================================

This example demonstrates how to build and train a CNN for image classification
using NovaNN with the MNIST dataset. Showcases the complete pipeline from data
loading to model evaluation.

Key concepts:
- Conv2d layers with different kernel sizes
- Batch normalization for training stability
- MaxPooling for spatial downsampling
- Global average pooling before classifier
- Learning rate scheduling
- Model checkpointing with validation
"""

import nova
import nova.nn as nn
import nova.nn.functional as F
from nova.optim import AdamW
from nova.optim.lr_scheduler import CosineAnnealingLR
from nova.metrics import Accuracy
from nova.utils.datasets import load_mnist_data
from nova.utils.data import DataLoader
from nova.utils.logger import logger


# Model Definition


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for MNIST classification.

    Architecture:
        Conv2d(1, 16) -> BN -> ReLU -> MaxPool
        Conv2d(16, 32) -> BN -> ReLU -> MaxPool
        Conv2d(32, 64) -> BN -> ReLU -> GlobalAvgPool
        Linear(64, 10)
    """

    def __init__(self):
        super().__init__()

        # Conv block 1: 1 -> 16 channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        # Conv block 2: 16 -> 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        # Conv block 3: 32 -> 64 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.global_pool = nn.GlobalAvgPool2d()

        # Flatten
        self.flatten = nn.Flatten()

        # Classifier
        self.classifier = nn.Linear(64, 10)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Block 1: 28x28 -> 14x14
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2: 14x14 -> 7x7
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3: 7x7 -> 1x1
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.global_pool(x)

        x = self.flatten(x)

        # Classifier
        x = self.classifier(x)
        return x


# Training & Evaluation


def train_epoch(model, loader, optimizer):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0

    for input, target in loader:
        optimizer.zero_grad()

        logits = model(input)
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, metric):
    """Evaluate model on given data."""
    model.eval()

    metric.reset()
    total_loss = 0.0

    with nova.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)

            metric.update(logits, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader), metric.compute().item()


# Main Training Pipeline


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 0.001

    print("=" * 70)
    logger.info("Convolutional Neural Network - MNIST Classification")
    print("=" * 70)

    # Load MNIST dataset
    logger.info("Loading MNIST dataset...")
    train_dataset, test_dataset, val_dataset = load_mnist_data(
        tensor4d=True,  # (N, 1, 28, 28) for CNN
        as_tensor=True,  # Convert to nova.Tensor
        do_normalize=True,  # Normalize to [0, 1]
        dtype=nova.float32,
    )

    logger.info(
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, optimizer, scheduler
    model = SimpleCNN()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    logger.info(f"\nModel:\n{model}")
    logger.info(f"\nTotal parameters: {sum(p.data.size for p in model.parameters()):,}")
    logger.info(f"Optimizer:\n{optimizer}")
    logger.info(f"Scheduler: CosineAnnealingLR (T_max={EPOCHS})")

    # Metrics
    train_metric = Accuracy(num_classes=10)
    val_metric = Accuracy(num_classes=10)

    # Training loop
    best_val_acc = 0.0
    print("\n" + "=" * 70)
    logger.info("Training started")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer)
        _, train_acc = evaluate(model, train_loader, train_metric)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, val_metric)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging
        logger.info(
            f"Epoch [{epoch:2d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            nova.save(model.state_dict(), "best_mnist_cnn.pth")
            logger.info(f"  → Best model saved (val_acc: {val_acc*100:.2f}%)")

    # Final evaluation on test set
    print("\n" + "=" * 70)
    logger.info("Evaluating on test set...")
    print("=" * 70)

    model.load_state_dict(nova.load("best_mnist_cnn.pth"))
    test_metric = Accuracy(num_classes=10)
    test_loss, test_acc = evaluate(model, test_loader, test_metric)

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    print("=" * 70)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
