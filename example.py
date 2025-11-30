import time
import numpy as np
from src.model import Sequential
from src.layers import Linear, ReLU, Conv2d, BatchNorm2d, Flatten, MaxPool2d

img = np.random.normal(0, 1, (5, 3, 64, 64))
nn = Sequential(
    Conv2d(3, 16, 3, padding=1),
    BatchNorm2d(16),
    ReLU(),
    Conv2d(16, 32, 3, stride=2),
    BatchNorm2d(32),
    ReLU(),
    MaxPool2d(2, 2),
    Flatten(),
    Linear(7200, 10),
)

start = time.perf_counter()
out = nn(img)
grad = np.zeros_like(out)
go = nn.backward(grad)
print(out.shape)
print(go.shape)
print(f"Execution time was {time.perf_counter() - start}s")
