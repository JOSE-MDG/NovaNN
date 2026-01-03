from __future__ import annotations
import numpy as np
from nova._interfaces._optimizer import Optimizer
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from nova.nn import Parameter
    from nova._typing import Closure


class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            parameters, {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        )
        self.eps = eps

    def _step_impl(self, closure: Closure = None) -> Optional[float]:
        loss = closure() if closure else None

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            wd = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                data = param.data

                state = self.state.setdefault(
                    param,
                    {
                        "step": 0,
                        "exp_avg": np.zeros_like(data, dtype=data.dtype),
                        "exp_avg_sq": np.zeros_like(data, dtype=data.dtype),
                    },
                )

                # increment step
                state["step"] += 1
                step = state["step"]

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # weight decay (coupled L2 regularization)
                if wd > 0 and not getattr(param, "is_bn_param", False):
                    grad += wd * data

                # moving averages
                m[:] = b1 * m + (1 - b1) * grad
                v[:] = b2 * v + (1 - b2) * (grad**2)

                # bias correction
                m_hat = m / (1 - b1**step)
                v_hat = v / (1 - b2**step)

                # parameter update
                denom = np.sqrt(v_hat) + self.eps
                param.data -= lr * (m_hat / denom)

        return loss
