from __future__ import annotations
import nova.nn.functional as F
from typing import TYPE_CHECKING
from nova.nn.modules import Module

if TYPE_CHECKING:
    from nova import Tensor


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self._training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Dropout2d(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout2d(input, self.p, self._training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Dropout3d(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout3d(input, self.p, self._training)

    def extra_repr(self) -> str:
        return f"p={self.p}"
