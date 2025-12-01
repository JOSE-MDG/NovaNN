# import numpy as np
# from novann.model import Sequential
# from novann.losses import CrossEntropyLoss
# from novann.metrics import accuracy
# from novann.core import DataLoader, logger
# from novann.optim import Adam
# from novann.layers import (
#     Linear,
#     ReLU,
#     Conv2d,
#     BatchNorm2d,
#     Flatten,
#     GlobalAvgPool2d,
#     BatchNorm1d,
#     Dropout,
# )

# np.random.seed(0)

# x_train, y_train = np.random.normal(0, 1, (10000, 3, 32, 32)), np.random.randint(
#     0, 10, 10000
# )
# x_test, y_test = np.random.normal(0, 1, (5000, 3, 32, 32)), np.random.randint(
#     0, 10, 5000
# )
# x_val, y_val = np.random.normal(0, 1, (5000, 3, 32, 32)), np.random.randint(0, 10, 5000)

# # Data loaders
# train_loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True)
# val_loader = DataLoader(x_val, y_val, batch_size=64, shuffle=False)
# test_loader = DataLoader(x_test, y_test, batch_size=64, shuffle=False)

# # Define Model
# model = Sequential(
#     Conv2d(3, 128, 5, stride=2, padding=2),
#     BatchNorm2d(128),
#     ReLU(),
#     Conv2d(128, 512, 5, stride=2, padding=2),
#     BatchNorm2d(512),
#     ReLU(),
#     GlobalAvgPool2d(),
#     Flatten(),
#     Linear(512, 64),
#     BatchNorm1d(64),
#     ReLU(),
#     Dropout(0.3),
#     Linear(64, 10),
# )

# # Hyperparameters
# learning_rate = 1e-3
# weight_decay = 1e-4
# optimizer = Adam(
#     model.parameters(),
#     learning_rate=learning_rate,
#     betas=(0.9, 0.999),
#     weight_decay=weight_decay,
#     epsilon=1e-8,
# )
# epochs = 5

# # Loss function
# loss_fn = CrossEntropyLoss()

# model.train()

# # Training loop
# for epoch in range(epochs):
#     losses = []
#     for input, label in train_loader:
#         # Set gradients to None
#         optimizer.zero_grad()

#         # Foward pass
#         logits = model(input)

#         # Compute loss and gradients
#         loss, grad = loss_fn(logits, label)
#         losses.append(loss)

#         # Backward pass
#         model.backward(grad)

#         # Update parameters
#         optimizer.step()

#     # Average losses per epoch
#     avg_losses = np.mean(losses)

#     # Validation accuracy after each epoch
#     model.eval()
#     acc = accuracy(model, val_loader)

#     model.train()
#     logger.info(
#         f"Epoch {epoch + 1}/{epochs}, Loss: {avg_losses:.4f}, Validation Accuracy: {acc:.4f}"
#     )

# # Final accuracy
# model.eval()
# accuracy = accuracy(model, test_loader)
# logger.info(f"Test Accuracy: {accuracy:.4f}", test_accuracy=round(accuracy))


from novann.core import DataLoader
import numpy as np

a, y = np.random.randn(1024, 3, 64, 64), np.random.randint(0, 2, 1024)

for x, y in DataLoader(a, y, 64, True):
    print(x.shape, y.shape)
