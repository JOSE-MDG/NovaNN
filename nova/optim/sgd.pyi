from __future__ import annotations
from typing import Iterable, Optional
from nova._interfaces._optimizer import Optimizer
from nova.nn import Parameter
from nova._typing import Closure

class SGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None: ...
    def _step_impl(self, closure: Closure = None) -> Optional[float]: ...
