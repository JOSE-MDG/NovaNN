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

                # init state if needed
                state = self.state.setdefault(
                    param,
                    {
                        "step": 0,
                        "exp_avg": np.zeros_like(data, dtype=data.dtype),
                        "exp_avg_sq": np.zeros_like(data, dtype=data.dtype),
                    },
                )

                # update state['step'] each step
                state["step"] += 1
                step = state["step"]

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # exponential moving averages
                m[:] = b1 * m + (1.0 - b1) * grad
                v[:] = b2 * v + (1.0 - b2) * (grad**2)

                # compute bias-corrected estimates
                bias_correction1 = 1.0 - (b1**step)
                bias_correction2 = 1.0 - (b2**step)

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # decoupled weight decay (apply to parameter before the adaptive step)
                if wd > 0 and not getattr(param, "is_bn_param", False):
                    data -= lr * wd * data
                    # write-back if needed
                    param.data = data

                # parameter update: lr * m_hat / (sqrt(v_hat) + eps)
                denom = np.sqrt(v_hat) + self.eps
                step_size = lr
                update = step_size * (m_hat / denom)

                # in-place subtract
                param.data -= update

        return loss
