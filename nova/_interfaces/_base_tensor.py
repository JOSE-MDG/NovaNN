from typing import TYPE_CHECKING
import numpy as np
import nova
from numpy import ndarray

if TYPE_CHECKING:
    from nova._typing import Dtype, Size


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
    def size(self) -> Size:
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

    def numel(self) -> int:
        return self._data_internal.size

    def dim(self) -> int:
        return self.ndim
