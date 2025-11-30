import time
import numpy as np
from src.model import Sequential
from src.layers import (
    Linear,
    ReLU,
    Conv2d,
    BatchNorm2d,
    BatchNorm1d,
    Flatten,
    MaxPool2d,
    Conv1d,
    GlobalAvgPool1d,
)

img4d = np.random.normal(0, 1, (5, 3, 64, 64))
img3d = np.random.normal(0, 1, (5, 3, 64))
nn2d = Sequential(
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

nn1d = Sequential(
    Conv1d(3, 16, 3, padding=1),
    BatchNorm1d(16),
    ReLU(),
    Conv1d(16, 32, 3, stride=2),
    BatchNorm1d(32),
    ReLU(),
    GlobalAvgPool1d(),
    Flatten(),
    Linear(32, 32),
    BatchNorm1d(32),
    ReLU(),
    Linear(32, 10),
)


start = time.perf_counter()
out = nn2d(img4d)
grad = np.zeros_like(out)
go = nn2d.backward(grad)
print(out.shape)
print(go.shape)
print(f"Execution time was {time.perf_counter() - start}s \n")

start = time.perf_counter()
out = nn1d(img3d)
grad = np.zeros_like(out)
go = nn1d.backward(grad)
print(out.shape)
print(go.shape)
print(f"Execution time was {time.perf_counter() - start}s")
