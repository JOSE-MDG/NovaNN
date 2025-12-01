import numpy as np
from novann.model import Sequential
from novann.losses import CrossEntropyLoss
from novann.metrics import accuracy
from novann.core import DataLoader, logger
from novann.optim import Adam
from novann.utils import load_mnist_data, train
from novann.layers import (
    Linear,
    ReLU,
    Conv2d,
    BatchNorm2d,
    Flatten,
    GlobalAvgPool2d,
    BatchNorm1d,
    Dropout,
)

np.random.seed(0)

(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_mnist_data(
    tensor4d=True, do_normalize=True
)

logger.info("training shape: {x_train.shape}")
logger.info("validation shape: {x_val.shape}")
logger.info("test shape: {x_test.shape}")

# Data loaders
train_loader = DataLoader(x_train, y_train, batch_size=256, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=256, shuffle=True)
test_loader = DataLoader(x_test, y_test, batch_size=256, shuffle=False)

# Define Model
model = Sequential(
    Conv2d(1, 256, 3, stride=3, padding=2),
    BatchNorm2d(128),
    ReLU(),
    Conv2d(256, 512, 3, stride=2, padding=1),
    BatchNorm2d(512),
    ReLU(),
    GlobalAvgPool2d(),
    Flatten(),
    Linear(512, 64),
    BatchNorm1d(64),
    ReLU(),
    Dropout(0.3),
    Linear(64, 10),
)

# Hyperparameters
epochs = 20
learning_rate = 0.003
weight_decay = 1e-5
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
logger.info(f"Test Accuracy: {accuracy:.4f}", test_accuracy=round(accuracy))
