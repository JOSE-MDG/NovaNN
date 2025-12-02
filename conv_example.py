import numpy as np
from novann.model import Sequential
from novann.losses import CrossEntropyLoss
from novann.metrics import accuracy
from novann.core import DataLoader, logger
from novann.optim import Adam
from novann.utils import train, load_mnist_data
from novann.layers import (
    Linear,
    ReLU,
    Conv2d,
    BatchNorm2d,
    Flatten,
    MaxPool2d,
    BatchNorm1d,
    Dropout,
)

np.random.seed(8)  # Established for reproducibility

# Load data
(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_mnist_data(
    tensor4d=True, do_normalize=True
)

# Shape validation
logger.info(f"training shape: {x_train.shape}")
logger.info(f"validation shape: {x_val.shape}")
logger.info(f"test shape: {x_test.shape}")

# Create loaders
train_loader = DataLoader(x_train, y_train, batch_size=256, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=256, shuffle=True)
test_loader = DataLoader(x_test, y_test, batch_size=256, shuffle=False)

# Create Model
model = Sequential(
    Conv2d(3, 128, 3, padding=1, bias=False),
    BatchNorm2d(128),
    ReLU(),
    MaxPool2d(2, 2),
    Conv2d(128, 256, 3, padding=1, bias=False),
    BatchNorm2d(256),
    ReLU(),
    MaxPool2d(2, 2),
    Flatten(),
    Linear(128, 64, bias=False),
    BatchNorm1d(64),
    ReLU(),
    Dropout(0.5),
    Linear(64, 10),
)

# Hyperparameters
epochs = 10
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = Adam(
    model.parameters(),
    learning_rate=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    epsilon=1e-8,
)

# Loss function
loss_fn = CrossEntropyLoss()

trained_model = train(
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
logger.info(f"Test Accuracy: {accuracy:.4f}", test_accuracy=round(accuracy, 4))
