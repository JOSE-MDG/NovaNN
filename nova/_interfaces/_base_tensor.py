from typing import Any, TYPE_CHECKING
from numpy import ndarray

if TYPE_CHECKING:
    from nova._typing import Dtype, Size
    from nova import Tensor


class TensorBase(Tensor):

    __slots__ = []

    @property
    def data(self) -> ndarray:
        return self._data_internal

    @data.setter
    def data(self, value: ndarray | Any):
        self._data_internal = value

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
