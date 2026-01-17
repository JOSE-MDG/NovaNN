from __future__ import annotations
from typing import Any
from numpy import ndarray

class Context:
    saved_tensors: tuple[ndarray, ...] | Any
    saved_shapes: tuple[tuple[int, ...], ...]

    def __init__(self) -> None: ...
    def save_for_backward(self, *args: ndarray) -> None: ...
