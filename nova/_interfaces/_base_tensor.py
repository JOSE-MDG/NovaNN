from typing import TYPE_CHECKING, Optional
import numpy as np
import nova
from numpy import ndarray

if TYPE_CHECKING:
    from nova._typing import Dtype, Size, Dim


class TensorBase:

    __slots__ = []

    @property
    def data(self) -> ndarray:
        return self._data_internal

    @data.setter
    def data(self, value: ndarray):
        if hasattr(value, "data") and isinstance(value, nova.Tensor):
            self._data_internal = value.data
        elif isinstance(value, np.ndarray):
            self._data_internal = value
        else:
            self._data_internal = np.asarray(value)

    @property
    def T(self):
        return self.permute()

    @property
    def shape(self) -> Size:
        """Returns the shape of the tensor (alias for shape)."""
        return self._data_internal.shape

    @property
    def dtype(self) -> Dtype:
        return self._dtype_internal

    @property
    def strides(self) -> tuple[int, ...]:
        return self._data_internal.strides

    @property
    def itemsize(self) -> int:
        return self._data_internal.itemsize

    @property
    def ndim(self) -> int:
        return self._data_internal.ndim

    @property
    def nbytes(self) -> int:
        return self._data_internal.nbytes

    @property
    def is_leaf(self) -> bool:
        """Returns True if this tensor is a leaf node."""
        return self._is_leaf

    @property
    def is_cuda(self) -> bool:
        """Returns False (Nova doesn't support CUDA yet)."""
        return False

    @property
    def device(self) -> str:
        """Returns 'cpu' (Nova only supports CPU)."""
        return "cpu"

    def numel(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self._data_internal.size

    def dim(self) -> int:
        return self.ndim

    def size(self, dim: Optional[Dim] = None) -> Dim:
        """
        Returns the size of the tensor.

        Args:
            dim: If specified, returns the size of that dimension.
                If None, returns the shape tuple.

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

        # handle negatives indices
        if dim < 0:
            dim = self.dim() + dim

        if dim < 0 or dim >= self.dim():
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{self.dim()}, {self.dim() -1}], but got {dim})"
            )

        return self.shape[dim]
