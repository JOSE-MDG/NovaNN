import numpy as np
from src.layers import BatchNorm1d, BatchNorm2d

data = np.random.randn(512, 64)
bn1 = BatchNorm1d(64)
out = bn1.forward(data)
print("BN1d: \n", out.mean(), out.std())

data2 = np.random.randn(64, 3, 32, 32)
bn2 = BatchNorm2d(3)
out2 = bn2.forward(data2)
print("BN2d: \n", out2.mean(), out2.std())

a = np.arange(4).reshape(2, 2)
print(np.pad(a, ((1, 1), (1, 1)), mode="wrap"))
