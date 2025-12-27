from __future__ import annotations
from typing import TYPE_CHECKING
from nova.utils import registry_class

if TYPE_CHECKING:
    from ._optimizer import Optimizer


@registry_class
class _LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer: Optimizer = optimizer
        self.last_epoch: int = last_epoch
        self.base_lrs: list[float] = [group["lr"] for group in optimizer.param_groups]
        self.step()

    def get_lr(self) -> list[float]:
        raise NotADirectoryError

    def step(self) -> None:

        self.last_epoch += 1

        new_lrs = self.get_lr()

        for group, lr in zip(self.optimizer.param_group, new_lrs):

            group["lr"] = lr

    def get_last_lr(self):

        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, list[float] | int]:
        return {
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, list[float] | int]) -> None:
        self.base_lrs = state_dict["base_lrs"]
        self.last_epoch = state_dict["last_epoch"]
