import numpy as np
import novann as nn
import novann.optim as optim

from novann.utils.data import DataLoader
from novann.utils.log_config import logger
from novann.utils.train import train
from novann.metrics import r2_score

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

np.random.seed(8)  # Established for reproducibility

x_data, y_data = make_regression(
    n_samples=40000, n_features=76, n_targets=1, noise=10.0, random_state=8
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
model = nn.Sequential(
    nn.Linear(76, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 16),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(16, 1),
)

# dataloaders
train_loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=64, shuffle=True)
test_loader = DataLoader(x_test, y_test, batch_size=64, shuffle=False)

# Hyperparameters
learning_rate = 4e-4
weight_decay = 1e-2
epochs = 50
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    epsilon=1e-8,
)

# Training
model = train(
    train_loader=train_loader,
    eval_loader=val_loader,
    net=model,
    optimizer=optimizer,
    loss_fn=nn.MSE(),
    epochs=epochs,
    show_logs_every=5,
    metric=r2_score,
)

# Final score
score = r2_score(model, test_loader)
logger.info(f"Test R²: {score:.4f}", score_r2=round(score))
