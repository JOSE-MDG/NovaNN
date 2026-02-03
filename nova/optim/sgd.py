"""
SGD optimizer for NovaNN.

Supports momentum and optional weight decay.
"""

from __future__ import annotations
import nova
import numpy as np
from nova._interfaces._optimizer import Optimizer
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from nova.nn import Parameter
    from nova._typing import Closure


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.

    Args:
        parameters (Iterable[Parameter]): Iterable of parameters to optimize.
        lr (float): Learning rate.
        momentum (float): Momentum factor. Defaults to 0.0.
        weight_decay (float): Weight decay (L2 penalty). Defaults to 0.0.

    Examples:
        >>> import nova
        >>> import numpy as np
        >>> from nova.nn import Parameter
        >>> from nova.optim import SGD
        >>>
        >>> p = Parameter(nova.randn(2, 2))
        >>> optimizer = SGD([p], lr=0.1, momentum=0.9)
        >>> for step in range(3):
        ...     p.grad = np.random.randn(*p.shape)
        ...     optimizer.step()
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Initialize SGD optimizer.

        Args:
            parameters (Iterable[Parameter]): Iterable of parameters to optimize.
            lr (float): Learning rate.
            momentum (float, optional): Momentum factor. Defaults to 0.0.
            weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.0.
        """

        super().__init__(
            params=parameters,
            defaults={"lr": lr, "weight_decay": weight_decay, "momentum": momentum},
        )

    def _step_impl(self, closure: Closure = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with nova.enable_grad():
                loss = closure()

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
                    grad += wd * data

                v = state["velocity"]

                if momentum > 0:
                    v[:] = momentum * v + grad
                    param.data -= lr * v
                else:
                    param.data -= lr * grad

        return loss
