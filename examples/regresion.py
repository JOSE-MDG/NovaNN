import numpy as np
import src.losses.functional as F

from src.core.logger import logger
from src.core.dataloader import DataLoader
from src.model import Sequential
from src.optim import SGD
from src.layers import Linear, Dropout, LeakyReLU, BatchNormalization
from src.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

x_data, y_data = make_regression(
    n_samples=40000, n_features=45, n_targets=1, noise=10.0, random_state=8
)

# Split data in train/val/test
x_train, x_test_val, y_train, y_test_val = train_test_split(
    x_data, y_data, test_size=0.3
)

x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=0.5)

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Define the regresor
model = Sequential(
    Linear(45, 368),
    BatchNormalization(368),
    LeakyReLU(),
    Dropout(0.2),
    Linear(368, 368),
    BatchNormalization(368),
    LeakyReLU(),
    Dropout(0.3),
    Linear(368, 176),
    BatchNormalization(176),
    LeakyReLU(),
    Dropout(0.4),
    Linear(176, 1),
)

# dataloaders
training_loader = DataLoader(x_train, y_train, batch_size=256, shuffle=True)
validation_loader = DataLoader(x_val, y_val, batch_size=256, shuffle=False)
test_loader = DataLoader(x_test, y_test, batch_size=256, shuffle=False)

# Hyperparameters
learning_rate = 1e-2
weight_decay = 1e-5
epochs = 50
optimizer = SGD(
    model.parameters(),
    learning_rate=learning_rate,
    momentum=0.9,
    weight_decay=weight_decay,
    max_grad_norm=1.0,
)
loss_fn = F.MSE()

# Training
model.train()
for epoch in range(epochs):
    losses = []
    for input, target in training_loader:
        # Set gradients to None
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input)

        # Compute loss
        loss, grad = loss_fn(outputs, target)

        # Backward pass
        model.backward(grad)

        # Update paramters
        optimizer.step()
        losses.append(loss)

    # Average training loss
    avg_losses = np.mean(losses)

    # Compute validation
    model.eval()
    r2 = r2_score(model, validation_loader)

    model.train()
    if (epoch + 1) % 5 == 0:
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_losses:.4f}, R²: {r2:.4f}")

# Final score
score = r2_score(model, test_loader)
logger.info(f"Test R²: {score:.4f}", score_r2=round(score))
