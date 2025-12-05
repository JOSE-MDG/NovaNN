import numpy as np
from typing import Iterator, Tuple


class DataLoader:
    """Iterable data loader that yields mini-batches from a dataset.

    This class wraps feature and label arrays and provides an iterator that
    return (xb, yb) batches. It supports shuffling the data at the start of
    each epoch.

    Attributes:
        x (np.ndarray): Input feature array of shape (N, ...).
        y (np.ndarray): Label array of shape (N, ...).
        bs (int): Batch size (i.e., number of samples per batch).
        shuffle (bool): Whether to shuffle samples each epoch.
    """

    class _Iter:
        """Iterator for DataLoader for a single epoch.

        This internal iterator maintains a shuffled index order (if enabled)
        and the current index within the epoch.

        Args:
            parent (DataLoader): Parent DataLoader instance.
        """

        def __init__(self, parent: "DataLoader"):
            self.parent: "DataLoader" = parent
            self.idx: int = 0
            # Create an array of indices. Shuffle it if required.
            self.order: np.ndarray = (
                np.random.permutation(len(parent.x))
                if parent.shuffle
                else np.arange(len(parent.x))
            )

        def __iter__(self) -> "DataLoader._Iter":
            """Return the iterator itself."""
            return self

        def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
            """Return the next batch (xb, yb).

            Raises:
                StopIteration: when the epoch is finished.
            """
            if self.idx >= len(self.parent.x):
                raise StopIteration

            i = self.idx
            j = min(self.idx + self.parent.bs, len(self.parent.x))
            batch_idx = self.order[i:j]

            xb = self.parent.x[batch_idx]
            yb = self.parent.y[batch_idx]

            self.idx = j
            return xb, yb

    def __init__(
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True
    ) -> None:
        """Initialize DataLoader.

        Args:
            x (np.ndarray): Feature array.
            y (np.ndarray): Label array.
            batch_size (int): Samples per batch. Defaults to 128.
            shuffle (bool): Shuffle each epoch. Defaults to True.
        """
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.bs: int = batch_size
        self.shuffle: bool = shuffle

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Return a new iterator for one epoch."""
        return DataLoader._Iter(self)

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        n = len(self.x)
        if n == 0:
            return 0
        return (n + self.bs - 1) // self.bs

    @property
    def batch_size(self) -> int:
        """Public read-only alias for the batch size."""
        return self.bs
