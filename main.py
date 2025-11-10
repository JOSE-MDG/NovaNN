import numpy as np
import pandas as pd
import src.model.nn as nn
import src.losses.Functional as F
from src.layers.linear.linear import Linear
from src.layers.activations.sigmoid import Sigmoid
from src.layers.activations.relu import LeakyReLU
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
    Linear(784, 200),
    BatchNormalization(200),
    LeakyReLU(),
    Dropout(0.5),
    Linear(200, 128),
    BatchNormalization(128),
    LeakyReLU(),
    Dropout(0.5),
    Linear(128, 64),
    BatchNormalization(64),
    LeakyReLU(),
    Dropout(0.5),
    Linear(64, 10),
)

optimizer = Adam(net.parameters(), 3e-2, betas=(0.9, 0.999), weight_decay=0.001)

net.train()
for epoch in range(20):
    for xb, yb in dl:

        logits = net(xb)

        loss = F.CrossEntropyLoss()
        cost, grad = loss(logits, yb)

        net.backward(grad)

        optimizer.step()
        net.zero_grad()

    acc = accuracy(net, dl)
    if epoch % 2 == 0:
        print(f"Loss: {round(cost, 3)} Acc: {acc*100:.2f}%")
