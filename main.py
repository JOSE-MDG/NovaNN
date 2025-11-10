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
from src.utils import accuracy


data = pd.read_csv(
    "/home/juancho_col/Documents/Neural Network/data/Mnist/MNIST_Train_e.csv"
)
x = data.drop(columns=["6"]).values
y = data["6"].values


dl = DataLoader(x, y, batch_size=1024)

net = nn.Sequential(
    Linear(784, 200, bias=False),
    BatchNormalization(200),
    ReLU(),
    Dropout(0.4),
    Linear(200, 128, bias=False),
    BatchNormalization(128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 64, bias=False),
    BatchNormalization(64),
    ReLU(),
    Dropout(0.5),
    Linear(64, 10, bias=False),
)

optimizer = Adam(net.parameters(), 4e-3, betas=(0.9, 0.999))
loss_fn = F.CrossEntropyLoss()

net.train()
for epoch in range(50):
    for xb, yb in dl:
        optimizer.zero_grad()
        logits = net(xb)
        cost, grad = loss_fn(logits, yb)
        net.backward(grad)
        optimizer.step()

    acc = accuracy(net, dl)
    if epoch % 2 == 0:
        print(f" - Epoch: {epoch} - Loss: {round(cost, 3)} - Acc: {acc*100:.2f}%")
