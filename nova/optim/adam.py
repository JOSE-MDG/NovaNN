"""
Adam optimizer for NovaNN.

Implements standard Adam algorithm with optional weight decay (L2 regularization).
"""

from __future__ import annotations
import nova
import numpy as np
from nova._interfaces._optimizer import Optimizer
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from nova.nn import Parameter
    from nova._typing import Closure


class Adam(Optimizer):
    """
    Standard Adam optimizer.

    Args:
        parameters (Iterable[Parameter]): Iterable of parameters to optimize.
        lr (float): Learning rate.
        betas (tuple[float, float]): Coefficients for running averages. Defaults to (0.9, 0.999).
        weight_decay (float): Weight decay (L2 penalty). Defaults to 0.0.
        eps (float): Small constant for numerical stability. Defaults to 1e-8.

    Examples:
        >>> import nova
        >>> import numpy as np
        >>> from nova.nn import Parameter
        >>> from nova.optim import Adam
        >>>
        >>> p = Parameter(nova.randn(3, 3))
        >>> optimizer = Adam([p], lr=0.01, weight_decay=0.01)
        >>> for step in range(3):
        ...     p.grad = np.random.randn(*p.shape)
        ...     optimizer.step()
    """

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
        loss = None
        if closure is not None:
            with nova.enable_grad():
                loss = closure()

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

                state["step"] += 1
                step = state["step"]

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # weight decay applied to gradient (coupled)
                if wd > 0 and not getattr(param, "is_bn_param", False):
                    grad += wd * data

                # exponential moving averages
                m[:] = b1 * m + (1 - b1) * grad
                v[:] = b2 * v + (1 - b2) * (grad**2)

                # bias correction
                m_hat = m / (1 - b1**step)
                v_hat = v / (1 - b2**step)

                # parameter update
                param.data -= lr * (m_hat / (np.sqrt(v_hat) + self.eps))

        return loss
