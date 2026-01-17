from __future__ import annotations
from typing import Optional
from numpy import ndarray
from nova import Tensor

def _build_topo(self: Tensor) -> list[Tensor]: ...
def _backward(
    cls: Tensor,
    gradient: Optional[ndarray | Tensor] = None,
    retain_graph: bool = False,
) -> None: ...
