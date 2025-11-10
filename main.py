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
from src.optim.rmsprop import RMSProp


data = pd.read_csv(
    "/home/juancho_col/Documents/Neural Network/data/Mnist/MNIST_Test.csv"
)
x = data.drop(columns=["7"]).values
y = data["7"].values


dl = DataLoader(x, y, batch_size=1024)

net = nn.Sequential(
    Linear(784, 200),
    BatchNormalization(200),
    LeakyReLU(),
    Dropout(0.5),
    Linear(200, 10),
)

optimizer = RMSProp(net.parameters(), 1e-2, beta=0.9, weight_decay=0.001)

net.train()
for epoch in range(20):
    for xb, yb in dl:

        logits = net(xb)

        loss = F.CrossEntropyLoss()
        cost, grad = loss(logits, yb)

        net.backward(grad)

        optimizer.step()
        nn.zero_grad()

    if epoch % 5 == 0:
        print("Loss:", cost)
