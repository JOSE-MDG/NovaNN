from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar
from ._optimizer import Optimizer

if TYPE_CHECKING:
    from nova._typing import SchedulerStateDict

TScheduler = TypeVar("TScheduler", bound="_LRScheduler")

class _LRScheduler:
    optimizer: Optimizer
    last_epoch: int
    base_lrs: list[float]

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None: ...
    def get_lr(self) -> list[float]: ...
    def step(self) -> None: ...
    def get_last_lr(self) -> list[float]: ...
    def state_dict(self) -> SchedulerStateDict: ...
    def load_state_dict(self, state_dict: SchedulerStateDict) -> None: ...
