from __future__ import annotations
import nova
from typing import Iterator, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from nova import Tensor


class DataLoader:
    """
    Iterable data loader that yields mini-batches from feature and label tensors.

    This class wraps input features and labels and provides an iterator that
    returns `(xb, yb)` batches for training or evaluation. Supports optional
    shuffling at the start of each epoch.

    Attributes:
        x (Tensor): Input feature tensor of shape (N, ...).
        y (Tensor): Label tensor of shape (N, ...).
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset each epoch.

    Examples:
        >>> import nova
        >>> from nova.utils.data import DataLoader
        >>> x = nova.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> y = nova.tensor([0, 1, 0])
        >>> loader = DataLoader(x, y, batch_size=2, shuffle=False)
        >>> for xb, yb in loader:
        ...     print(xb, yb)
        tensor([[1., 2.],
                [3., 4.]]) tensor([0., 1.])
        tensor([[5., 6.]]) tensor([0.])
    """

    class _Iter:
        """Iterator for DataLoader for a single epoch.

        This internal iterator maintains a shuffled index order (if enabled)
        and the current index within the epoch.

        Args:
            parent (DataLoader): Parent DataLoader instance.
        """

        def __init__(self, parent: DataLoader):
            self.parent: DataLoader = parent
            self.idx: int = 0
            # Create an array of indices. Shuffle it if required.
            self.order: Tensor = (
                nova.randperm(len(parent.x), dtype=nova.long)
                if parent.shuffle
                else nova.arange(len(parent.x), dtype=nova.long)
            )

        def __iter__(self) -> DataLoader._Iter:
            """Return the iterator itself."""
            return self

        def __next__(self) -> Tuple[Tensor, Tensor]:
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
        self, x: Tensor, y: Tensor, batch_size: int = 64, shuffle: bool = True
    ) -> None:
        """Initialize DataLoader.

        Args:
            x (Tensor): Feature tensor.
            y (Tensor): Label tensor.
            batch_size (int): Samples per batch. Defaults to 128.
            shuffle (bool): Shuffle each epoch. Defaults to True.
        """
        self.x: Tensor = x
        self.y: Tensor = y
        self.bs: int = batch_size
        self.shuffle: bool = shuffle

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
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
