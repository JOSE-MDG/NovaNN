import pytest
import numpy as np
from src.core.dataloader import DataLoader

x_data = np.random.randn(10, 4)
y_data = np.random.randint(0, 10, (10,))


def test_last_batch_size():
    loader = DataLoader(x_data, y_data, 4)
    batchs = list(loader)
    last_batch = batchs[-1]
    assert (len(last_batch[0]) and len(last_batch[1])) == 2
