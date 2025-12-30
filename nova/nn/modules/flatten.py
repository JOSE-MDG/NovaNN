from __future__ import annotations
import nova
import nova.nn.functional as F
from typing import TYPE_CHECKING
from nova.nn.modules import Module
from nova.nn.parameter import Parameter

if TYPE_CHECKING:
    from nova import Tensor


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim: int = start_dim
        self.end_dim: int = end_dim

    def forward(self, input: Tensor) -> Tensor:
        return F.flatten(input, start_dim=self.start_dim, end_dim=self.end_dim)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"
