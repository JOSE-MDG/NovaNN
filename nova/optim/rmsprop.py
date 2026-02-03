"""
RMSprop optimizer for NovaNN.

Supports optional momentum, centered variance, and weight decay.
"""

from __future__ import annotations
import nova
import numpy as np
from nova._interfaces._optimizer import Optimizer
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from nova.nn import Parameter
    from nova._typing import Closure


class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    Args:
        parameters (Iterable[Parameter]): Iterable of parameters to optimize.
        lr (float): Learning rate.
        alpha (float): Smoothing constant. Defaults to 0.99.
        weight_decay (float): L2 penalty. Defaults to 0.0.
        momentum (float): Momentum factor. Defaults to 0.0.
        centered (bool): Whether to normalize by the centered variance. Defaults to True.
        eps (float): Small term for numerical stability. Defaults to 1e-8.

    Examples:
        >>> import nova
        >>> import numpy as np
        >>> from nova.nn import Parameter
        >>> from nova.optim import RMSprop
        >>>
        >>> p = Parameter(nova.randn(2, 2))
        >>> optimizer = RMSprop([p], lr=0.01, alpha=0.9)
        >>> for step in range(3):
        ...     p.grad = np.random.randn(*p.shape)
        ...     optimizer.step()
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float,
        alpha: float = 0.99,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = True,
        eps: float = 1e-8,
    ) -> None:
        """
        Initialize RMSprop optimizer.

        Args:
            parameters (Iterable[Parameter]): Iterable of parameters to optimize.
            lr (float): Learning rate.
            alpha (float, optional): Smoothing constant for the moving average of squared gradients.
                Defaults to 0.99.
            weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.0.
            momentum (float, optional): Momentum factor. Defaults to 0.0.
            centered (bool, optional): If True, computes the centered RMSprop by normalizing
                gradients by an estimate of their variance. Defaults to True.
            eps (float, optional): Term added to the denominator to improve numerical stability.
                Defaults to 1e-8.
        """

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
        self.eps = eps

    def _step_impl(self, closure: Closure = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with nova.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            centered = group["centered"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                data = param.data

                state = self.state.setdefault(
                    param,
                    {
                        "step": 0,
                        "exp_avg_sq": np.zeros_like(data, dtype=data.dtype),
                        "exp_avg": np.zeros_like(data, dtype=data.dtype),
                        "velocity": np.zeros_like(data, dtype=data.dtype),
                    },
                )

                state["step"] += 1

                if wd > 0 and not getattr(param, "is_bn_param", False):
                    grad += wd * data

                state["exp_avg_sq"][:] = alpha * state["exp_avg_sq"] + (1 - alpha) * (
                    grad**2
                )

                if centered:
                    state["exp_avg"][:] = alpha * state["exp_avg"] + (1 - alpha) * grad
                    variance = state["exp_avg_sq"] - state["exp_avg"] ** 2
                    variance = np.maximum(variance, 1e-20)
                    denom = np.sqrt(variance) + self.eps
                else:
                    denom = np.sqrt(state["exp_avg_sq"]) + self.eps

                if momentum > 0:
                    state["velocity"][:] = momentum * state["velocity"] + grad / denom
                    update = state["velocity"]
                else:
                    update = grad / denom

                param.data -= lr * update

        return loss
