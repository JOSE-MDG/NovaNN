from __future__ import annotations
import numpy as np
from nova._interfaces._optimizer import Optimizer
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from nova.nn import Parameter
    from nova._typing import Closure


class RMSprop(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float,
        alpha: float = 0.99,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = True,
    ):
        super().__init__(
            parameters,
            {
                "lr": lr,
                "alpha": alpha,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "centered": centered,
            },
        )

        self.eps = 1e-8

    def _step_impl(self, closure: Closure = None) -> Optional[float]:
        loss = closure() if closure is not None else None

        for group in self.param_groups:

            lr = group["lr"]
            alpha = group["alpha"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            centered = group["centered"]

            for param in group["params"]:

                state = self.state.setdefault(
                    param,
                    {
                        "exp_avg_sq": np.zeros_like(param.data, dtype=param.dtype),
                        "exp_avg": np.zeros_like(param.data),
                        "velocity": np.zeros_like(param.data, dtype=param.dtype),
                    },
                )

                # 1. weight decay
                if wd > 0 and not getattr(param, "is_bn_param", False):
                    param.grad += wd * param.data

                # 2. update exp avg mean

                state["exp_avg_sq"][:] = alpha * state["exp_avg_sq"] + (1 - alpha) * (
                    param.grad**2
                )

                # 3. gradient normalization
                if centered:
                    state["exp_avg"][:] = (
                        alpha * state["exp_avg"] + (1 - alpha) * param.grad
                    )
                    safe_var = np.maximum(
                        state["exp_avg_sq"] - (state["exp_avg"] ** 2), 1e-20
                    )
                    denom = np.sqrt(safe_var) + self.eps
                else:
                    denom = np.sqrt(state["exp_avg_sq"]) + self.eps

                if momentum > 0:
                    state["velocity"][:] = momentum * state["velocity"] + (
                        param.grad / denom
                    )
                else:
                    state["velocity"][:] = param.grad / denom

                # 5. final Update
                param.data -= lr * state["velocity"]

        return loss
