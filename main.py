"""
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
net = nn.Sequential()

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
epochs = 5
batch_size = 1024

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
    acc = accuracy(net, val_loader)
    if epoch % 2 == 0:
        logger.info(
            f"Epoch {epoch + 1}/{epochs}, Loss: {cost:.4f}, Validation Accuracy: {acc:.4f}"
        )
    accuracy_history.append(acc)
    loss_history.append(cost)


# Test the model
test_accuracy = accuracy(net, test_loader)
logger.info(f"Test Accuracy: {test_accuracy:.4f}")

# Save training history
history = {"accuracy": accuracy_history, "loss": loss_history}
with open("training_history.json", "w") as f:
    json.dump(history, f)
logger.info("Training history saved to 'training_history.json'")
"""
