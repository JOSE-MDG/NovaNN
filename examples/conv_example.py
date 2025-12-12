import numpy as np
import novann as nn
import novann.optim as optim

from novann.metrics import accuracy
from novann.utils.train import train
from novann.utils.data import DataLoader
from novann.utils.log_config import logger
from novann.utils.datasets import load_mnist_data

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
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, 3, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 64, bias=False),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.35),
    nn.Linear(64, 10),
)

# Hyperparameters
epochs = 10
learning_rate = 1e-3
weight_decay = 1e-2
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    epsilon=1e-8,
)

# Loss function
loss_fn = nn.CrossEntropyLoss()

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
