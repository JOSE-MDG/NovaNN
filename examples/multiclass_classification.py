import numpy as np
import novann as nn
import novann.optim as optim

from novann.utils.data import DataLoader
from novann.utils.train import train
from novann.utils.datasets import load_mnist_data
from novann.utils.log_config import logger
from novann.metrics import accuracy

np.random.seed(8)  # Established for reproducibility

# Load data
(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_mnist_data()

# Data loaders
train_loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=64, shuffle=True)
test_loader = DataLoader(x_test, y_test, batch_size=64, shuffle=False)

# Define Model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 10),
)

# Hyperparameters
learning_rate = 1e-3
weight_decay = 1e-2
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    epsilon=1e-8,
)
epochs = 50

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Training loop
model = train(
    train_loader=train_loader,
    eval_loader=val_loader,
    net=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=epochs,
    metric=accuracy,
)

# Final accuracy
model.eval()
accuracy = accuracy(model, test_loader)
logger.info(f"Test Accuracy: {accuracy:.4f}", test_accuracy=round(accuracy))
