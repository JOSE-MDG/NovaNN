import json
import src.losses.functional as F

from src.core.logger import logger
from src.core.dataloader import DataLoader
from src.layers import Linear, ReLU, BatchNormalization, Dropout
from src.model.nn import Sequential
from src.optim import Adam
from src.utils import load_fashion_mnist_data, accuracy

# History records
accuracy_history = []
loss_history = []

# Load data
(x_train, y_train), (x_test, y_test), (x_val, y_val) = (
    load_fashion_mnist_data()
)  # (50K, 784), (10K, 784), (10K, 784) samples

# Data loaders
train_loader = DataLoader(x_train, y_train, batch_size=256, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=256, shuffle=False)
test_loader = DataLoader(x_test, y_test, batch_size=256, shuffle=False)

# Define model
net = Sequential(
    # Layer 1
    Linear(28 * 28, 512, bias=False),
    BatchNormalization(512),
    ReLU(),
    Dropout(0.3),
    # Layer 2
    Linear(512, 256, bias=False),
    BatchNormalization(256),
    ReLU(),
    Dropout(0.3),
    # Layer 3
    Linear(256, 128, bias=False),
    BatchNormalization(128),
    ReLU(),
    Dropout(0.2),
    # Output Layer
    Linear(128, 10, bias=False),
)

# prepare for training
net.train()

# Hyperparameters
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = Adam(
    net.parameters(),
    learning_rate=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    epsilon=1e-8,
)
epochs = 50
batch_size = 256

# Loss function
loss_fn = F.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    for input, label in train_loader:
        # Set gradients to None
        optimizer.zero_grad()

        # Foward pass
        logits = net(input)

        # Compute loss and gradients
        cost, grad = loss_fn(logits, label)

        # Backward pass
        net.backward(grad)

        # Update parameters
        optimizer.step()

    # Validation accuracy after each epoch
    net.eval()
    acc = accuracy(net, val_loader)

    net.train()
    if (epoch + 1) % 5 == 0:
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Loss: {cost:.4f}, Validation Accuracy: {acc:.4f}"
        )
    accuracy_history.append(acc)
    loss_history.append(cost)


# Test the model
net.eval()
test_accuracy = accuracy(net, test_loader)
logger.info(f"Test Accuracy: {test_accuracy:.4f}", final_accuracy=round(test_accuracy))

# Save training history within a JSON file for later comparison
history = {"accuracy": accuracy_history, "loss": loss_history}
with open("training_history.json", "w") as f:
    json.dump(history, f)
logger.info(
    "Training history saved to `training_history.json` ",
    history_file="training_history.json",
)
