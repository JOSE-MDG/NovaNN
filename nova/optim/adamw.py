from __future__ import annotations
import numpy as np
from nova._interfaces._optimizer import Optimizer
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from nova.nn import Parameter
    from nova._typing import Closure


class AdamW(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__(
            params=parameters,
            defaults={"lr": lr, "betas": betas, "weight_decay": weight_decay},
        )

        self.t = 0
        self.eps = eps

    def _step_impl(self, closure: Closure = None) -> Optional[float]:

        self.t += 1

        loss = closure() if closure else None

        for group in self.param_groups:

            lr = group["lr"]
            b1 = group["betas"][0]
            b2 = group["betas"][1]
            wd = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state.setdefault(
                    param,
                    {
                        "step": self.t,
                        "exp_avg": np.zeros_like(param.data, dtype=param.dtype),
                        "exp_avg_sq": np.zeros_like(param.data, dtype=param.dtype),
                    },
                )

                state["exp_avg"] = b1 * state["exp_avg"] + (1 - b1) * param.grad
                state["exp_avg_sq"] = b2 * state["exp_avg_sq"] + (1 - b2) * (
                    param.grad**2
                )

                bias_correction1 = state["exp_avg"] / (1 - b1 ** state["step"])
                bias_correction2 = state["exp_avg_sq"] / (1 - b2 ** state["step"])

                param.data -= (
                    lr * bias_correction1 / (np.sqrt(bias_correction2 + self.eps))
                )

                if wd > 0 and not getattr(param, "is_bn_param", False):
                    param.data -= lr * wd * param.data

        return loss
