"""
Learning rate schedulers for NovaNN.

Contains StepLR, CosineAnnealingLR, and OneCycleLR schedulers.
"""

from __future__ import annotations
import math
from nova._interfaces._lr_scheduler import _LRScheduler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova._interfaces._optimizer import Optimizer

__all__ = ["StepLR", "CosineAnnealingLR", "OneCycleLR"]


class StepLR(_LRScheduler):
    """
    Step learning rate scheduler.

    Decays the learning rate of each parameter group by `gamma` every `step_size` steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay. Default: 1.0.
        last_epoch (int): The index of last epoch. Default: -1.

    Examples:
        >>> import nova
        >>> import numpy as np
        >>> from nova.optim import SGD
        >>> from nova.nn import Parameter
        >>>
        >>> p = Parameter(nova.randn(2, 2))
        >>> optimizer = SGD([p], lr=0.1)
        >>> scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
        >>> for epoch in range(5):
        ...     p.grad = np.random.randn(*p.shape)
        ...     optimizer.step()
        ...     scheduler.step()
        ...     print(f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']}")
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize StepLR scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            step_size (int): Period of learning rate decay.
            gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 1.0.
            last_epoch (int, optional): The index of last epoch. Defaults to -1.
        """
        self.step_size: int = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [lr * factor for lr in self.base_lrs]


class CosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing learning rate scheduler.

    Gradually decreases learning rate following a cosine curve from base_lr to eta_min.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of steps.
        eta_min (float): Minimum learning rate. Default: 0.0.
        last_epoch (int): Index of last epoch. Default: -1.

    Examples:
        >>> import nova
        >>> import numpy as np
        >>> from nova.optim import SGD
        >>> from nova.nn import Parameter
        >>>
        >>> p = Parameter(nova.randn(2, 2))
        >>> optimizer = SGD([p], lr=0.1)
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0.01)
        >>> for step in range(6):
        ...     p.grad = np.random.randn(*p.shape)
        ...     optimizer.step()
        ...     scheduler.step()
        ...     print(f"Step {step}, lr: {optimizer.param_groups[0]['lr']}")
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize CosineAnnealingLR scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_max (int): Maximum number of iterations/steps.
            eta_min (float, optional): Minimum learning rate. Defaults to 0.0.
            last_epoch (int, optional): The index of last epoch. Defaults to -1.
        """
        self.T_max: int = T_max
        self.eta_min: float = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        progress = min(self.last_epoch / self.T_max, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs
        ]


class OneCycleLR(_LRScheduler):
    """
    One-cycle learning rate scheduler.

    Cycles learning rate from initial_lr -> max_lr -> final_lr over `total_steps` steps.
    Optionally cycles momentum inversely.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float): Maximum learning rate.
        total_steps (int): Total number of steps in the cycle.
        pct_start (float): Fraction of cycle spent increasing LR. Default: 0.3.
        div_factor (float): Initial LR = max_lr / div_factor. Default: 25.0.
        final_div_factor (float): Minimum LR = max_lr / final_div_factor. Default: 1e4.
        cycle_momentum (bool): Whether to adjust momentum inversely. Default: True.
        max_momentum (float): Maximum momentum during cycle. Default: 0.95.
        last_epoch (int): Last epoch index. Default: -1.

    Examples:
        >>> import nova
        >>> import numpy as np
        >>> from nova.optim import SGD
        >>> from nova.nn import Parameter
        >>>
        >>> p = Parameter(np.random.randn(2, 2))
        >>> optimizer = SGD([p], lr=0.1, momentum=0.9)
        >>> scheduler = OneCycleLR(optimizer, max_lr=0.2, total_steps=5)
        >>> for step in range(5):
        ...     p.grad = np.random.randn(*p.shape)
        ...     optimizer.step()
        ...     scheduler.step()
        ...     print(f"Step {step}, lr: {optimizer.param_groups[0]['lr']}, momentum: {optimizer.param_groups[0]['momentum']}")
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        cycle_momentum: bool = True,
        max_momentum: float = 0.95,
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize OneCycleLR scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            max_lr (float): Upper learning rate boundary in the cycle.
            total_steps (int): Total number of steps in the cycle.
            pct_start (float, optional): Percentage of the cycle spent increasing the learning rate.
                Defaults to 0.3.
            div_factor (float, optional): Determines the initial learning rate via
                initial_lr = max_lr / div_factor. Defaults to 25.0.
            final_div_factor (float, optional): Determines the minimum learning rate via
                min_lr = max_lr / final_div_factor. Defaults to 1e4.
            cycle_momentum (bool, optional): If True, momentum is cycled inversely to learning rate.
                Defaults to True.
            max_momentum (float, optional): Upper momentum boundary in the cycle. Defaults to 0.95.
            last_epoch (int, optional): The index of the last epoch. Defaults to -1.
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start

        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor

        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up

        self.cycle_momentum = cycle_momentum
        self.max_momentum = max_momentum

        if "momentum" in optimizer.param_groups[0]:
            self.base_momentums = [
                group["momentum"] for group in optimizer.param_groups
            ]
            self.momentum_type = "momentum"
        elif "betas" in optimizer.param_groups[0]:
            self.base_momentums = [
                group["betas"][0] for group in optimizer.param_groups
            ]
            self.momentum_type = "betas"
        else:
            self.base_momentums = None
            self.momentum_type = None
            self.cycle_momentum = False

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch <= self.step_up:
            pct = self.last_epoch / self.step_up
            distance = pct * (self.max_lr - self.initial_lr)
            lr = self.initial_lr + distance
        else:
            pct = (self.last_epoch - self.step_up) / self.step_down
            cosine = (1 + math.cos(math.pi * pct)) / 2
            distance = self.max_lr - self.final_lr
            lr = self.final_lr + distance * cosine

        return [lr for _ in self.base_lrs]

    def get_momentum(self) -> list[float]:
        if not self.cycle_momentum:
            return self.base_momentums

        if self.last_epoch <= self.step_up:
            pct = self.last_epoch / self.step_up
            return [
                self.max_momentum - pct * (self.max_momentum - base_m)
                for base_m in self.base_momentums
            ]
        else:
            pct = (self.last_epoch - self.step_up) / self.step_down
            cosine = (1 + math.cos(math.pi * pct)) / 2
            return [
                base_m + (self.max_momentum - base_m) * cosine
                for base_m in self.base_momentums
            ]

    def step(self) -> None:
        self.last_epoch += 1

        lrs = self.get_lr()
        moms = self.get_momentum()

        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = lrs[i]
            if self.cycle_momentum:
                if self.momentum_type == "momentum":
                    group["momentum"] = moms[i]
                elif self.momentum_type == "betas":
                    group["betas"] = (moms[i], group["betas"][1])
