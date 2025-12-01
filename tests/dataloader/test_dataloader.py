import pytest
import numpy as np
from novann.core.dataloader import DataLoader

# Small deterministic dataset for the test (shape: 10 x 4)
x_data = np.random.randn(10, 4)
y_data = np.random.randint(0, 10, (10,))


def test_last_batch_size():
    """Ensure DataLoader yields the final (smaller) batch correctly.

    With batch_size=4 and 10 samples we expect batches of sizes: 4, 4, 2.
    The test checks the last yielded batch has 2 samples.
    """
    loader = DataLoader(x_data, y_data, 4)
    batchs = list(loader)
    last_batch = batchs[-1]
    # last_batch is a tuple (X_batch, y_batch) — verify final batch length is 2
    assert (len(last_batch[0]) and len(last_batch[1])) == 2
