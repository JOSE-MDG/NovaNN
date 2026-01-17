from __future__ import annotations
import nova
from .dataset import Dataset
from typing import Iterator, Tuple, TYPE_CHECKING, Self


if TYPE_CHECKING:
    from nova import Tensor


class DataLoader:
    """Iterable data loader that yields mini-batches from a Dataset.

    This class wraps a Dataset and provides an iterator that returns ``(xb, yb)``
    batches for training or evaluation. It supports optional shuffling at the start
    of each epoch and handles batch creation automatically.

    The DataLoader is a fundamental component of the training pipeline, enabling
    efficient iteration over datasets with configurable batch sizes and shuffling.
    It works seamlessly with any Dataset subclass.

    Args:
        dataset: Dataset instance to load data from
        batch_size: Number of samples per batch. Default: 64
        shuffle: Whether to shuffle the dataset indices at the start of each epoch.
            Default: ``True``

    Attributes:
        dataset (Dataset): The dataset to load from
        bs (int): Internal storage for batch size
        shuffle (bool): Whether shuffling is enabled

    Examples::

        >>> # Basic usage with a custom dataset
        >>> class MyDataset(Dataset):
        ...     def __init__(self, size):
        ...         self.data = nova.randn(size, 10)
        ...         self.labels = nova.randint(0, 2, (size,))
        ...
        ...     def __len__(self):
        ...         return len(self.data)
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[index], self.labels[index]
        ...
        >>> dataset = MyDataset(100)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for xb, yb in loader:
        ...     print(xb.shape, yb.shape)
        ...     break
        (32, 10) (32,)

        >>> # Training loop example
        >>> model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))
        >>> optimizer = SGD(model.parameters(), lr=0.01)
        >>> loss_fn = CrossEntropyLoss()
        >>>
        >>> for epoch in range(10):
        ...     for xb, yb in loader:
        ...         pred = model(xb)
        ...         loss = loss_fn(pred, yb)
        ...         loss.backward()
        ...         optimizer.step()
        ...         optimizer.zero_grad()

        >>> # Without shuffling (for evaluation)
        >>> test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        >>> model.eval()
        >>> for xb, yb in test_loader:
        ...     pred = model(xb)
        ...     # compute metrics

        >>> # Check number of batches
        >>> print(len(loader))  # Number of batches per epoch
        4  # (100 samples / 32 batch_size = 4 batches)

        >>> # Access batch_size property
        >>> print(loader.batch_size)
        32
    """

    class _Iter:
        """Iterator for DataLoader for a single epoch.

        This internal iterator maintains a shuffled or sequential index order
        and tracks the current position within the epoch. It handles batching
        by slicing the index order and retrieving samples from the parent dataset.

        Args:
            parent: Parent DataLoader instance containing the dataset and configuration

        Attributes:
            parent (DataLoader): Reference to the parent DataLoader
            idx (int): Current index position in the epoch
            order (Tensor): Tensor of indices defining the iteration order (shuffled or sequential)
        """

        def __init__(self, parent: DataLoader) -> None:
            self.parent: DataLoader = parent
            self.idx: int = 0
            # Create an array of indices. Shuffle it if required.
            self.order: Tensor = (
                nova.randperm(len(parent.dataset), dtype=nova.long)
                if parent.shuffle
                else nova.arange(len(parent.dataset), dtype=nova.long)
            )

        def __iter__(self) -> Self:
            """Returns the iterator itself.

            Returns:
                Self reference for iterator protocol
            """
            return self

        def __next__(self) -> Tuple[Tensor, Tensor]:
            """Returns the next batch (xb, yb).

            Retrieves a batch of samples by slicing the index order and using
            the indices to fetch data from the parent dataset. Automatically
            handles the last batch which may be smaller than batch_size.

            Returns:
                Tuple[Tensor, Tensor]: A tuple containing:
                    - xb: Batch of input samples
                    - yb: Batch of corresponding labels

            Raises:
                StopIteration: When all samples in the epoch have been yielded

            Examples::

                >>> loader = DataLoader(dataset, batch_size=32)
                >>> iterator = iter(loader)
                >>> xb, yb = next(iterator)  # First batch
                >>> xb2, yb2 = next(iterator)  # Second batch
            """
            if self.idx >= len(self.parent.dataset):
                raise StopIteration

            i = self.idx
            j = min(self.idx + self.parent.bs, len(self.parent.dataset))
            batch_idx = self.order[i:j]

            xb, yb = self.parent.dataset[batch_idx]

            self.idx = j
            return xb, yb

    def __init__(
        self, dataset: Dataset, batch_size: int = 64, shuffle: bool = True
    ) -> None:
        """Initializes the DataLoader.

        Args:
            dataset: Dataset instance implementing ``__len__`` and ``__getitem__``
            batch_size: Number of samples per batch. Must be positive. Default: 64
            shuffle: Whether to shuffle dataset indices at the start of each epoch.
                Useful for training (True) but typically disabled for evaluation (False).
                Default: ``True``

        Examples::

            >>> # Training dataloader with shuffling
            >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            >>> # Evaluation dataloader without shuffling
            >>> val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            >>> # Small batch size for limited memory
            >>> loader = DataLoader(dataset, batch_size=8, shuffle=True)
        """
        self.dataset = dataset
        self.bs: int = batch_size
        self.shuffle: bool = shuffle

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Returns a new iterator for one epoch.

        Creates a fresh iterator each time, ensuring proper shuffling (if enabled)
        at the start of each epoch. This allows the DataLoader to be used in
        multiple epochs with independent shuffling.

        Returns:
            Iterator that yields batches of (input, target) tuples

        Examples::

            >>> loader = DataLoader(dataset, batch_size=32)
            >>> # Each call to iter() creates a new iterator with fresh shuffling
            >>> for epoch in range(3):
            ...     for xb, yb in loader:  # New iterator each epoch
            ...         # training code
            ...         pass
        """
        return DataLoader._Iter(self)

    def __len__(self) -> int:
        """Returns the number of batches per epoch.

        Calculates how many batches are needed to cover all samples in the dataset
        given the current batch size. The last batch may contain fewer samples than
        batch_size if the dataset size is not evenly divisible.

        Returns:
            Number of batches that will be yielded in one complete epoch

        Examples::

            >>> dataset = MyDataset(100)
            >>> loader = DataLoader(dataset, batch_size=32)
            >>> print(len(loader))
            4  # ceil(100 / 32) = 4 batches

            >>> # Last batch will have 4 samples (100 - 3*32 = 4)
            >>> loader = DataLoader(dataset, batch_size=32)
            >>> batches = list(loader)
            >>> print(batches[-1][0].shape[0])  # Last batch size
            4

            >>> # Empty dataset
            >>> empty_dataset = MyDataset(0)
            >>> empty_loader = DataLoader(empty_dataset, batch_size=32)
            >>> print(len(empty_loader))
            0
        """
        n = len(self.dataset)
        if n == 0:
            return 0
        return (n + self.bs - 1) // self.bs

    @property
    def batch_size(self) -> int:
        """Returns the batch size as a read-only property.

        Provides public access to the batch size without allowing modification.
        This is useful for logging, debugging, or when other components need to
        know the batch size being used.

        Returns:
            The configured batch size

        Examples::

            >>> loader = DataLoader(dataset, batch_size=32)
            >>> print(f"Using batch size: {loader.batch_size}")
            Using batch size: 32

            >>> # Read-only: cannot be modified
            >>> # loader.batch_size = 64  # This would raise an AttributeError
        """
        return self.bs
