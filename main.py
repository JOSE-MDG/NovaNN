import numpy as np
import pandas as pd
import src.model.nn as nn
import src.losses.Functional as F
from src.layers.linear.linear import Linear
from src.layers.activations.sigmoid import Sigmoid
from src.layers.activations.relu import LeakyReLU, ReLU
from src.layers.regularization.dropout import Dropout
from src.layers.bn.batch_normalization import BatchNormalization
from src.core.dataloader import DataLoader
from src.optim.adam import Adam
from src.optim.rmsprop import RMSProp
from src.utils import accuracy


data_train = pd.read_csv(
    "/home/juancho_col/Documents/Neural Network/data/Mnist/MNIST_Train_e.csv"
)

data_test = pd.read_csv(
    "/home/juancho_col/Documents/Neural Network/data/Mnist/MNIST_Test.csv"
)

x = data_train.drop(columns=["6"]).values
y = data_train["6"].values

x_test = data_test.drop(columns=["7"]).values
y_test = data_test["7"].values

loader_train = DataLoader(x, y, batch_size=512)
loader_test = DataLoader(x_test, y_test, batch_size=512)

net = nn.Sequential(
    Linear(784, 636, bias=False),
    BatchNormalization(636),
    ReLU(),
    Dropout(0.5),
    Linear(636, 174, bias=False),
    BatchNormalization(174),
    ReLU(),
    Dropout(0.5),
    Linear(174, 10, bias=False),
)

epochs = 50
learning_rate = 0.01
optimizer1 = Adam(net.parameters(), learning_rate, betas=(0.9, 0.999))
optimizer2 = RMSProp(net.parameters(), learning_rate, beta=0.9)

loss_fn = F.CrossEntropyLoss()

net.train()

print(f"First train with Adam optimizer:\n")
for epoch in range(epochs):
    for xb, yb in loader_train:
        optimizer1.zero_grad()
        logits = net(xb)
        cost, grad = loss_fn(logits, yb)
        net.backward(grad)
        optimizer1.step()

    acc = accuracy(net, loader_test)
    if epoch % 5 == 0:
        print(f"(Adam) - Epoch: {epoch} - Loss: {round(cost, 3)} - Acc: {acc*100:.3f}%")

print("")

print(f"Second train with RMSProp optimizer:\n")
for epoch in range(epochs):
    for xb, yb in loader_train:
        optimizer2.zero_grad()
        logits = net(xb)
        cost, grad = loss_fn(logits, yb)
        net.backward(grad)
        optimizer1.step()

    acc = accuracy(net, loader_test)
    if epoch % 5 == 0:
        print(
            f"(RMSProp) - Epoch: {epoch} - Loss: {round(cost, 3)} - Acc: {acc*100:.3f}%"
        )
