from __future__ import annotations
from nova._interfaces._lr_scheduler import _LRScheduler
from nova._interfaces._optimizer import Optimizer

class StepLR(_LRScheduler):
    step_size: int
    gamma: float

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ) -> None: ...
    def get_lr(self) -> list[float]: ...

class CosineAnnealingLR(_LRScheduler):
    T_max: int
    eta_min: float

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None: ...
    def get_lr(self) -> list[float]: ...

class OneCycleLR(_LRScheduler):
    max_lr: float
    total_steps: int
    pct_start: float
    initial_lr: float
    final_lr: float
    step_up: int
    step_down: int
    cycle_momentum: bool
    max_momentum: float
    base_momentums: list[float] | None
    momentum_type: str | None

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
    ) -> None: ...
    def get_lr(self) -> list[float]: ...
    def get_momentum(self) -> list[float]: ...
    def step(self) -> None: ...
