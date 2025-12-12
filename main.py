import json
import novann as nn
import novann.optim as optim
from novann.utils.data import DataLoader
from novann.utils.datasets import load_mnist_data
from novann.utils.log_config import logger
from novann.metrics import accuracy

# History records
accuracy_history = []
loss_history = []

# Load data
(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_mnist_data(
    tensor4d=True, do_normalize=True
)

# Data loaders
train_loader = DataLoader(x_train, y_train, batch_size=128, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=128, shuffle=True)
test_loader = DataLoader(x_test, y_test, batch_size=128, shuffle=False)

# Define model
model = nn.Sequential(
    # Block 1: 28x28x1 -> 28x28x16
    nn.Conv2d(1, 16, 3, padding=1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    # Block 2: 28x28x16 -> 14x14x32
    nn.Conv2d(16, 32, 3, padding=1, stride=2, bias=False),  # Stride for Downsampling
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
    # Block 3: 7x7x32 -> 7x7x64
    nn.Conv2d(32, 64, 3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    # classification layers
    nn.GlobalAvgPool2d(),  # 7x7x64 -> 1x1x64 (Global Average Pooling)
    nn.Flatten(),  # Vector of 64 elements
    nn.Linear(64, 32, bias=False),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(32, 10),  # 10 classes
)

count = 0
for p in model.parameters():
    count += p.data.size

logger.info(f"Model Strcuture: \n\n {model} \n")
logger.info(f"The model have {count} parametrs")

# prepare for training
model.train()

# Hyperparameters
lr = 1e-3
weight_decay = 1e-2
optimizer = optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    epsilon=1e-8,
)
epochs = 50

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    for input, label in train_loader:
        # Set gradients to zero
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

    # Save results
    accuracy_history.append(round(acc, 4))
    loss_history.append(round(loss, 4))

    model.train()
    if (epoch + 1) % 5 == 0:
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}"
        )

# Test the model
model.eval()
test_accuracy = accuracy(model, test_loader)
logger.info(
    f"Test Accuracy: {test_accuracy:.4f}", final_accuracy=round(test_accuracy, 4)
)

# Save training history within a JSON file for later comparison
history = {"accuracy": accuracy_history, "loss": loss_history}

with open("./training_history.json", "w") as f:
    json.dump(history, f)
print("Training history saved as `pytorch_training_history.json` ")

logger.info(
    "Training history saved to `training_history.json` ",
    history_file="training_history.json",
)
