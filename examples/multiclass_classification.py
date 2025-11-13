import src.losses.functional as F

from src.core.logger import logger
from src.core.dataloader import DataLoader
from src.layers import Linear, ReLU, BatchNormalization, Dropout
from src.model.nn import Sequential
from src.optim import Adam
from src.utils import load_mnist_data
from src.metrics import accuracy

# Load data
(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_mnist_data()

# Data loaders
train_loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=64, shuffle=False)
test_loader = DataLoader(x_test, y_test, batch_size=64, shuffle=False)

# Define Model
model = Sequential(
    Linear(784, 256),
    BatchNormalization(256),
    ReLU(),
    Dropout(0.3),
    Linear(256, 128),
    BatchNormalization(128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 64),
    BatchNormalization(64),
    ReLU(),
    Dropout(0.2),
    Linear(64, 10),
)

# Hyperparameters
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = Adam(
    model.parameters(),
    learning_rate=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    epsilon=1e-8,
)
epochs = 50

# Loss function
loss_fn = F.CrossEntropyLoss()

model.train()

# Training loop
for epoch in range(epochs):
    for input, label in train_loader:
        # Set gradients to None
        optimizer.zero_grad()

        # Foward pass
        logits = model(input)

        # Compute loss and gradients
        loss, grad = loss_fn(logits, label)

        # Backward pass
        model.backward(grad)

        # Update parameters
        optimizer.step()

    # Validation accuracy after each epoch
    model.eval()
    acc = accuracy(model, val_loader)

    model.train()
    if (epoch + 1) % 5 == 0:
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}"
        )

# Final accuracy
model.eval()
accuracy = accuracy(model, test_loader)
logger.info(f"Test Accuracy: {accuracy:.4f}", test_accuracy=round(accuracy))
