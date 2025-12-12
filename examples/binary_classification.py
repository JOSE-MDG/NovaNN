import numpy as np
import novann as nn
import novann.optim as optim

from novann.utils.data import DataLoader
from novann.utils.train import train
from novann.utils.log_config import logger
from novann.metrics import binary_accuracy

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(8)  # Established for reproducibility

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
model = nn.Sequential(
    nn.Linear(2, 32),
    nn.Tanh(),
    nn.Dropout(0.25),
    nn.Linear(32, 16),
    nn.Tanh(),
    nn.Dropout(0.33),
    nn.Linear(16, 4),
    nn.Tanh(),
    nn.Dropout(0.2),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)

# Hyperparameters
epochs = 30
learning_rate = 1e-3
weight_decay = 1e-2

optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    lambda_l1=True,
)

# Loss function
loss_fn = nn.BinaryCrossEntropy()

# DataLoaders
train_loader = DataLoader(x=x_train, y=y_train)
val_loader = DataLoader(x=x_val, y=y_val)
test_dataloader = DataLoader(x=x_test, y=y_test)

# Training
model = train(
    train_loader=train_loader,
    eval_loader=val_loader,
    net=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=epochs,
    metric=binary_accuracy,
)

# Final accuracy
accuracy = binary_accuracy(model, test_dataloader)
logger.info(f"Test Accuracy: {accuracy:.4f}", test_accuracy=round(accuracy))
