import numpy as np
import src.losses.functional as F
from src.core.logger import logger
from src.core.dataloader import DataLoader
from src.model.nn import Sequential
from src.layers import Linear, Sigmoid, Tanh, Dropout
from src.metrics import binary_accuracy
from src.optim import Adam
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Create data
x_binary, y_binary = make_moons(n_samples=40000, noise=0.1, random_state=8)
y_binary = y_binary.reshape(-1, 1)

# Split data in train/val/test
x_train, x_test_val, y_train, y_test_val = train_test_split(
    x_binary, y_binary, test_size=0.2, stratify=y_binary
)

x_test, x_val, y_test, y_val = train_test_split(
    x_test_val, y_test_val, test_size=0.5, stratify=y_test_val
)

# Define model
model = Sequential(
    Linear(2, 32),
    Tanh(),
    Dropout(0.2),
    Linear(32, 16),
    Tanh(),
    Dropout(0.2),
    Linear(16, 4),
    Tanh(),
    Dropout(0.2),
    Linear(4, 1),
    Sigmoid(),
)

# Hyperparameters
epochs = 30
learning_rate = 1e-2
weight_decay = 1e-5

optimizer = Adam(
    model.parameters(),
    learning_rate=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    lambda_l1=True,
)

# Loss function
loss_fn = F.BinaryCrossEntropy()

# DataLoaders
training_dataloader = DataLoader(x=x_train, y=y_train)
validation_dataloader = DataLoader(x=x_val, y=y_val)
test_dataloader = DataLoader(x=x_test, y=y_test)

# Training
model.train()
for epoch in range(epochs):
    losses = []
    for input, target in training_dataloader:
        # Set gradients to None
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input)

        # Compute loss
        loss, grad = loss_fn(outputs, target)
        losses.append(loss)

        # Backward pass
        model.backward(grad)

        # Update paramters
        optimizer.step()

    # Average losses per epoch
    avg_losses = np.mean(losses)

    # Compute validation accuracy
    model.eval()
    acc = binary_accuracy(model, validation_dataloader)

    model.train()
    if (epoch + 1) % 5 == 0:
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_losses:.4f}, Validation Accuracy: {acc:.4f}"
        )

# Final accuracy
accuracy = binary_accuracy(model, test_dataloader)
logger.info(f"Test Accuracy: {accuracy:.4f}", test_accuracy=round(accuracy))
