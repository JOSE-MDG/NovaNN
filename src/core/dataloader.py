import numpy as np

from typing import Tuple


class DataLoader:
    """A data loader that provides an iterable over a dataset.

    This class takes a dataset (features `x` and labels `y`) and makes it
    iterable in mini-batches. It supports shuffling the data at the beginning
    of each epoch.
    """

    class _Iter:
        """Internal iterator class for the DataLoader.

        This class holds the state of the iteration for one epoch, including
        the current index and the shuffled order of the data.
        """

        def __init__(self, parent: "DataLoader"):
            """Initializes the iterator.

            Args:
                parent (DataLoader): The DataLoader instance to iterate over.
            """
            self.parent = parent
            self.idx = 0
            # Create an array of indices. Shuffle it if required.
            self.order = (
                np.random.permutation(len(parent.x))
                if parent.shuffle
                else np.arange(len(parent.x))
            )

        def __iter__(self):
            """Returns the iterator object itself."""
            return self

        def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
            """Returns the next batch from the dataset.

            Raises:
                StopIteration: When all batches have been yielded for the epoch.

            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing the batch of
                                               features (xb) and labels (yb).
            """
            # Stop iteration if the index has gone past the end of the dataset
            if self.idx >= len(self.parent.x):
                raise StopIteration

            # Get the start and end indices for the current batch
            i = self.idx
            j = min(self.idx + self.parent.bs, len(self.parent.x))
            batch_idx = self.order[i:j]

            # Slice the dataset to create the batch
            xb = self.parent.x[batch_idx]
            yb = self.parent.y[batch_idx]

            # Advance the index for the next iteration
            self.idx = j

            return xb, yb

    def __init__(
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, shuffle: bool = True
    ):
        """Initializes the DataLoader.

        Args:
            x (np.ndarray): The input features.
            y (np.ndarray): The corresponding labels.
            batch_size (int): The number of samples per batch. Defaults to 128.
            shuffle (bool): Whether to shuffle the data at each epoch. Defaults to True.
        """
        self.x = x
        self.y = y
        self.bs = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        """Returns a new iterator object for the dataset for one epoch."""
        return DataLoader._Iter(self)
