import json
import src.model.nn as nn
import src.losses.functional as F

from src.core.logger import logger
from src.core.dataloader import DataLoader
from src.layers.linear.linear import Linear
from src.layers.activations.relu import ReLU
from src.layers.regularization.dropout import Dropout
from src.layers.bn.batch_normalization import BatchNormalization
from src.optim.adam import Adam
from src.utils import load_fashion_mnist_data, accuracy

# History records
accuracy_history = []
loss_history = []

# Load data
(x_train, y_train), (x_test, y_test), (x_val, y_val) = (
    load_fashion_mnist_data()
)  # (50K, 784), (10K, 784), (10K, 784) samples

# Data loaders
train_loader = DataLoader(x_train, y_train, batch_size=128, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=128, shuffle=False)
test_loader = DataLoader(x_test, y_test, batch_size=128, shuffle=False)

# Define model
net = nn.Sequential(
    Linear(28 * 28, 1024),
    BatchNormalization(1024),
    ReLU(),
    Dropout(0.3),
    Linear(1024, 512),
    BatchNormalization(512),
    ReLU(),
    Dropout(0.3),
    Linear(512, 256),
    BatchNormalization(256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    BatchNormalization(128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 10),
)

# Hyperparameters
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = Adam(
    net.parameters(),
    learning_rate=learning_rate,
    betas=(0.9, 0.99),
    weight_decay=weight_decay,
    epsilon=1e-8,
)
epochs = 30
batch_size = 128

# Loss function
loss_fn = F.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = net(xb)
        cost, grad = loss_fn(logits, yb)
        net.backward(grad)
        optimizer.step()

        # Validation
        acc = accuracy(net, val_loader)
        accuracy_history.append(acc)
        loss_history.append(cost)

    if (epoch + 1) % 2 == 0:
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {cost:.4f}, Validation Accuracy: {acc:.4f}"
        )

# Test the model
test_accuracy = accuracy(net, test_loader)
logger.info(f"Test Accuracy: {test_accuracy:.4f}")

# Save training history
history = {"accuracy": accuracy_history, "loss": loss_history}
with open("training_history.json", "w") as f:
    json.dump(history, f)
logger.info("Training history saved to 'training_history.json'")
