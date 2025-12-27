from __future__ import annotations
from typing import Any
from numpy import ndarray


class Context:
    def __init__(self):
        self.saved_tensors: tuple[ndarray, ...] | Any = ()
        self.saved_shapes: tuple[tuple[int, ...], ...] = ()

    def save_for_backward(self, *args: ndarray):
        self.saved_tensors = args
