from typing import Iterable, Optional
from nova._interfaces._optimizer import Optimizer
from nova.nn import Parameter
from nova._typing import Closure

class RMSprop(Optimizer):
    eps: float

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float,
        alpha: float = 0.99,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = True,
        eps: float = 1e-8,
    ) -> None: ...
    def _step_impl(self, closure: Closure = None) -> Optional[float]: ...
