import numpy as np
import src.model.nn as nn
from src.layers.linear.linear import Linear
from src.layers.activations.sigmoid import Sigmoid
from src.layers.activations.relu import ReLU
from src.layers.regularization.dropout import Dropout
from src.layers.bn.batch_normalization import BatchNormalization

F = 12
S = 512

x = np.random.randn(F, S)

seq = nn.Sequential(
    Linear(12, 200),
    BatchNormalization(200),
    ReLU(),
    Dropout(0.5),
    Linear(200, 10),
    BatchNormalization(10),
    Sigmoid(),
)

activations, _ = seq._find_next_activation(0)

out = seq.forward(x)
