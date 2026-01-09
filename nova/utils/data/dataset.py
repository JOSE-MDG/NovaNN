from __future__ import annotations
from typing import TYPE_CHECKING
from numpy import ndarray

if TYPE_CHECKING:
    from nova import Tensor

# Type alias for dataset indexing
type Index = slice | int | tuple | Tensor | ndarray


class Dataset:
    """Abstract base class for all datasets.

    All datasets that represent a map from keys to data samples should subclass this.
    All subclasses must override ``__getitem__`` and ``__len__`` methods to support
    indexing and getting the size of the dataset respectively.

    Datasets are used in conjunction with DataLoaders to provide efficient batching
    and iteration over data during training and evaluation. By implementing the
    required methods, your dataset can be seamlessly integrated into the training
    pipeline.

    Examples::

        >>> # Creating a simple custom dataset
        >>> class MyDataset(Dataset):
        ...     def __init__(self, data, labels):
        ...         self.data = data
        ...         self.labels = labels
        ...
        ...     def __len__(self):
        ...         return len(self.data)
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[index], self.labels[index]
        ...
        >>> # Usage
        >>> data = nova.randn(100, 10)
        >>> labels = nova.randint(0, 2, (100,))
        >>> dataset = MyDataset(data, labels)
        >>> print(len(dataset))  # 100
        >>> sample, label = dataset[0]
        >>> print(sample.shape, label.shape)  # (10,) ()

        **Clarification**: Transformations are not currently
        incorporated into the framework; this is an
        example to reflect future versions.

        >>> # Dataset with data augmentation
        >>> class AugmentedDataset(Dataset):
        ...     def __init__(self, data, labels, transform=None):
        ...         self.data = data
        ...         self.labels = labels
        ...         self.transform = transform
        ...
        ...     def __len__(self):
        ...         return len(self.data)
        ...
        ...     def __getitem__(self, index):
        ...         sample = self.data[index]
        ...         if self.transform:
        ...             sample = self.transform(sample)
        ...         return sample, self.labels[index]
        ...
        >>> def normalize(x):
        ...     return (x - x.mean()) / x.std()
        >>> dataset = AugmentedDataset(data, labels, transform=normalize)

        >>> # Dataset for image data
        >>> class ImageDataset(Dataset):
        ...     def __init__(self, image_paths, labels):
        ...         self.image_paths = image_paths
        ...         self.labels = labels
        ...
        ...     def __len__(self):
        ...         return len(self.image_paths)
        ...
        ...     def __getitem__(self, index):
        ...         # Load image from path
        ...         image = load_image(self.image_paths[index])
        ...         label = self.labels[index]
        ...         return image, label

        >>> # Slicing support
        >>> dataset = MyDataset(data, labels)
        >>> batch = dataset[0:10]  # Get first 10 samples
        >>> samples = dataset[[1, 5, 9]]  # Get specific indices

    Note:
        When implementing ``__getitem__``, ensure it returns a tuple of (input, target)
        for supervised learning tasks. The exact format may vary based on your use case.
    """

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        This method must be implemented by all subclasses. It is used by DataLoaders
        to determine how many samples can be drawn from the dataset.

        Returns:
            int: The total number of samples in the dataset

        Raises:
            NotImplementedError: If subclass does not implement this method

        Examples::

            >>> class MyDataset(Dataset):
            ...     def __init__(self, size):
            ...         self.size = size
            ...
            ...     def __len__(self):
            ...         return self.size
            ...
            ...     def __getitem__(self, index):
            ...         return nova.randn(10), nova.tensor(0)
            ...
            >>> dataset = MyDataset(1000)
            >>> print(len(dataset))  # 1000
        """
        raise NotImplementedError(
            "Sub class of dataset should implement '__len__' method"
        )

    def __getitem__(self, index: Index) -> tuple[Tensor, Tensor]:
        """Retrieves a sample or batch of samples from the dataset.

        This method must be implemented by all subclasses. It should support integer
        indexing for single samples, as well as more advanced indexing with slices,
        tuples, tensors, or numpy arrays for batch access.

        Args:
            index: Index or indices to retrieve. Can be:
                - ``int``: Single sample index
                - ``slice``: Range of samples (e.g., ``dataset[0:10]``)
                - ``tuple``: Multiple indices
                - ``Tensor``: Tensor of indices
                - ``ndarray``: NumPy array of indices

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Input data tensor
                - Target/label tensor

        Raises:
            NotImplementedError: If subclass does not implement this method
            IndexError: If index is out of bounds

        Examples::

            >>> class MyDataset(Dataset):
            ...     def __init__(self, data, labels):
            ...         self.data = data
            ...         self.labels = labels
            ...
            ...     def __len__(self):
            ...         return len(self.data)
            ...
            ...     def __getitem__(self, index):
            ...         return self.data[index], self.labels[index]
            ...
            >>> data = nova.randn(100, 10)
            >>> labels = nova.randint(0, 5, (100,))
            >>> dataset = MyDataset(data, labels)

            >>> # Single sample
            >>> sample, label = dataset[0]
            >>> print(sample.shape)  # (10,)

            >>> # Slice
            >>> samples, labels = dataset[0:5]
            >>> print(samples.shape)  # (5, 10)

            >>> # List of indices
            >>> indices = [1, 5, 10]
            >>> samples, labels = dataset[indices]

            >>> # Tensor indices
            >>> indices = nova.tensor([0, 2, 4])
            >>> samples, labels = dataset[indices]
        """
        raise NotImplementedError(
            "Sub class of dataset should implement '__getitem__' method"
        )
