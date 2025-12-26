from __future__ import annotations
from numpy import ndarray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor


class Context:
    def __init__(self):
        self.saved_tensors: tuple[Tensor, ...] = ()
        self.saved_shapes: tuple[tuple[int, ...], ...] = ()

    def save_for_backward(self, *args: ndarray):
        self.saved_tensors = args
