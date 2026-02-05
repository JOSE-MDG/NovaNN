from typing import Iterable, Optional
from nova._interfaces._optimizer import Optimizer
from nova.nn import Parameter
from nova._typing import Closure

class AdamW(Optimizer):
    eps: float

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ) -> None: ...
    def _step_impl(self, closure: Closure = None) -> Optional[float]: ...
