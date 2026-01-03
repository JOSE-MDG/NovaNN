from __future__ import annotations
import numpy as np
from nova._interfaces._optimizer import Optimizer
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from nova.nn import Parameter
    from nova._typing import Closure


class SGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params=parameters,
            defaults={"lr": lr, "weight_decay": weight_decay, "momentum": momentum},
        )

    def _step_impl(self, closure: Closure = None) -> Optional[float]:
        loss = closure() if closure else None

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                data = param.data

                state = self.state.setdefault(
                    param, {"velocity": np.zeros_like(data, dtype=data.dtype)}
                )

                if wd > 0 and not getattr(param, "is_bn_param", False):
                    grad = grad + wd * data

                v = state["velocity"]

                if momentum > 0:
                    v[:] = momentum * v + grad
                    param.data -= lr * v
                else:
                    param.data -= lr * grad

        return loss
