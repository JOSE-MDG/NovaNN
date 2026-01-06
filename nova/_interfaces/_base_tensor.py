from typing import TYPE_CHECKING, Optional, Self
import numpy as np
import nova
from numpy import ndarray

if TYPE_CHECKING:
    from nova._typing import Dtype, Size, Dim


class TensorBase:
    """
    Base class providing core tensor properties and metadata access.

    This abstract base defines the interface for tensor shape, dtype, strides,
    and other fundamental attributes. Actual tensor operations are implemented
    in subclasses and in the yaml file.
    """

    __slots__ = []

    @property
    def data(self) -> ndarray:
        """Returns the underlying numpy array containing tensor data."""
        return self._data_internal

    @data.setter
    def data(self, value: ndarray) -> None:
        """
        Sets the underlying data from numpy array or Nova tensor.
        The tensor .data allways must be a ndarray.

        Args: value (ndarray): NumPy array that will act as the .data file of the Tensor wrapper class
        """
        if hasattr(value, "data") and isinstance(value, nova.Tensor):
            self._data_internal = value.data
        elif isinstance(value, np.ndarray):
            self._data_internal = value
        else:
            self._data_internal = np.asarray(value)

    @property
    def T(self) -> Self:
        """Returns the transposed tensor (convenience alias for permute())."""
        return self.permute()

    @property
    def shape(self) -> Size:
        """Returns the shape of the tensor as a tuple of dimensions."""
        return self._data_internal.shape

    @property
    def dtype(self) -> Dtype:
        """Returns the data type of the tensor elements."""
        return self._dtype_internal

    @property
    def strides(self) -> tuple[int, ...]:
        """Returns the stride of each dimension in bytes."""
        return self._data_internal.strides

    @property
    def itemsize(self) -> int:
        """Returns the size in bytes of each element."""
        return self._data_internal.itemsize

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions."""
        return self._data_internal.ndim

    @property
    def nbytes(self) -> int:
        """Returns the total bytes consumed by the tensor's elements."""
        return self._data_internal.nbytes

    @property
    def is_leaf(self) -> bool:
        """Returns True if this tensor is a leaf node in the computational graph."""
        return self._is_leaf

    @property
    def is_cuda(self) -> bool:
        """Returns False (NovaNN doesn't support CUDA yet)."""
        return False

    @property
    def device(self) -> str:
        """Returns 'cpu' (NovaNN only supports CPU currently)."""
        return "cpu"

    def numel(self) -> int:
        """
        Returns the total number of elements in the tensor.

        Examples:
            >>> x = nova.tensor([[1, 2], [3, 4]])
            >>> x.numel()
            4
        """
        return self._data_internal.size

    def dim(self) -> int:
        """Returns the number of dimensions (same as ndim)."""
        return self.ndim

    def size(self, dim: Optional[Dim] = None) -> Dim:
        """
        Returns the size of the tensor or a specific dimension.

        Args:
            dim: Dimension index. If None, returns full shape tuple.
                Supports negative indexing.

        Returns:
            Shape tuple if dim is None, otherwise size of the specified dimension.

        Examples:
            >>> x = nova.randn(2, 3, 4)
            >>> x.size()
            (2, 3, 4)
            >>> x.size(0)
            2
            >>> x.size(-1)
            4
        """
        if dim is None:
            return self.shape

        # Handle negative indices
        if dim < 0:
            dim = self.dim() + dim

        if dim < 0 or dim >= self.dim():
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[-{self.dim()}, {self.dim() - 1}], but got {dim})"
            )

        return self.shape[dim]
