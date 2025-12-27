from __future__ import annotations
import math
from nova._interfaces._lr_scheduler import _LRScheduler
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:

    from nova._interfaces._optimizer import Optimizer


class StepLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        self.step_size: int = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:

        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [lr * factor for lr in self.base_lrs]


class CosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.T_max: int = T_max
        self.eta_min: int = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = min(self.last_epoch / self.T_max, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs
        ]


class OneCycleLR(_LRScheduler):
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
        # last_epoch -> last_step
        if self.last_epoch <= self.step_up:

            pct = (
                self.last_epoch / self.step_up
            )  # It tells us how far we have progressed in the warm-up phase

            distance = pct * (
                self.max_lr
                - self.initial_lr  # (max_lr - initial_lr) -> total distance in the warm-up phase
            )  # distance from the entry point

            lr = self.initial_lr + distance
        else:
            # last_epoch -> last_step
            pct = (self.last_epoch - self.step_up) / self.step_down

            cosine = (
                1 + math.cos(math.pi * pct)
            ) / 2  # move the range from [1,-1] to [1,0]

            distance = (
                self.max_lr - self.final_lr
            )  # # (max_lt - final_lr) -> total distance in the cool-down phase

            lr = self.final_lr + distance * cosine

        return [lr for _ in self.base_lrs]

    def get_momentum(self) -> list[float]:
        if not self.cycle_momentum:
            return self.base_momentums

        # last_epoch -> last_step
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
