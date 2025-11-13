import numpy as np

from src.core.logger import logger
from src.core.dataloader import DataLoader
from src.model.nn import Sequential
from src.layers import Linear, Sigmoid, Tanh, Dropout
from src.utils import binary_accuracy
from src.losses import BinaryCrossEntropy
from src.optim import Adam
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Load data
x_binary, y_binary = make_moons(n_samples=40000, noise=0.1, random_state=8)
y_binary = y_binary.reshape(-1, 1)

# Split train/val/test data
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
loss_fn = BinaryCrossEntropy()

# DataLoaders
training_dataLoader = DataLoader(x=x_train, y=y_train)
validation_dataloader = DataLoader(x=x_val, y=y_val)
test_dataloader = DataLoader(x=x_test, y=y_test)

# Training
model.train()
for epoch in range(epochs):
    for input, target in training_dataLoader:
        # Set gradients to None
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input)

        # Compute loss
        probs = (outputs >= 0.5).astype(np.int64)
        loss, grad = loss_fn(probs, target)

        # Backward pass
        model.backward(grad)

        # Update paramters
        optimizer.step()

    # Compute validation accuracy
    model.eval()
    acc = binary_accuracy(model, validation_dataloader)

    model.train()
    if (epoch + 1) % 5 == 0:
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}"
        )
