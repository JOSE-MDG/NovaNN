from __future__ import annotations
from typing import TYPE_CHECKING
from nova.utils import registry_class

if TYPE_CHECKING:
    from ._optimizer import Optimizer
    from nova._typing import SchedulerStateDict


@registry_class
class _LRScheduler:
    """
    Base class for learning rate schedulers.

    A scheduler updates the learning rate of an optimizer according to
    a predefined schedule. Subclasses must implement `get_lr`.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        """
        Initialize the learning rate scheduler.

        Args:
            optimizer: Optimizer whose learning rate will be scheduled.
            last_epoch: Index of the last epoch. Use -1 for initial state.
        """
        self.optimizer: Optimizer = optimizer
        self.last_epoch: int = last_epoch
        self.base_lrs: list[float] = [group["lr"] for group in optimizer.param_groups]
        self.step()

    def get_lr(self) -> list[float]:
        """
        Compute learning rates for the current step.

        Subclasses must override this method.

        Returns:
            List of learning rates, one per parameter group.
        """
        raise NotADirectoryError

    def step(self) -> None:
        """
        Advance the scheduler by one step and update learning rates.

        Example:
            >>> optimizer = nova.optim.SGD(model.parameters(), lr=0.1)
            >>> scheduler = nova.optim.lr_scheduler.StepLR(optimizer, step_size=10)
            >>>
            >>> for epoch in range(100):
            ...     train(...)
            ...     scheduler.step()
        """
        self.last_epoch += 1

        new_lrs = self.get_lr()

        for group, lr in zip(self.optimizer.param_groups, new_lrs):

            group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        """
        Return the last computed learning rates.

        Returns:
            List of current learning rates.
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> SchedulerStateDict:
        """
        Return the scheduler state.

        Returns:
            A dictionary containing scheduler state.
        """
        return {
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: SchedulerStateDict) -> None:
        """
        Load the scheduler state.

        Args:
            state_dict: Scheduler state dictionary.
        """
        self.base_lrs = state_dict["base_lrs"]
        self.last_epoch = state_dict["last_epoch"]
