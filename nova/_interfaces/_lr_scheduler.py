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
            optimizer (Optimizer): Wrapped optimizer whose learning rate will be scheduled.
            last_epoch (int, optional): The index of the last epoch. Use -1 to start from
                the beginning. Defaults to -1.

        Note:
            Automatically calls step() once during initialization to set the initial
            learning rate.
        """
        self.optimizer: Optimizer = optimizer
        self.last_epoch: int = last_epoch
        self.base_lrs: list[float] = [group["lr"] for group in optimizer.param_groups]
        self.step()

    def __getstate__(self) -> SchedulerStateDict:
        """
        Prepare scheduler state for pickling.

        Returns:
            SchedulerStateDict: Dictionary containing the scheduler's state, excluding
                the optimizer reference to avoid circular dependencies during serialization.

        Note:
            The optimizer reference is excluded and must be restored manually after
            unpickling using load_state_dict().
        """
        state = self.__dict__.copy()
        state.pop("optimizer", None)
        return state

    def __setstate__(self, state: SchedulerStateDict) -> None:
        """
        Restore scheduler state after unpickling.

        Args:
            state (SchedulerStateDict): Dictionary containing the scheduler's state.

        Note:
            The optimizer must be set manually after unpickling. The scheduler will not
            be functional until an optimizer is assigned.

        Example:
            >>> import pickle
            >>> # Save
            >>> state = pickle.dumps(scheduler)
            >>> # Load
            >>> scheduler = pickle.loads(state)
            >>> scheduler.optimizer = optimizer  # Must restore optimizer reference
        """
        self.__dict__.update(state)

    def get_lr(self) -> list[float]:
        """
        Compute learning rates for the current step.

        Subclasses must override this method.

        Returns:
            List of learning rates, one per parameter group.
        """
        raise NotImplementedError

    def step(self) -> None:
        """
        Advance the scheduler by one step and update learning rates.

        Example:
            >>> import nova.optim as optim
            >>>
            >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
            >>> scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
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
            state_dict (SchedulerStateDict): Scheduler state dictionary.
        """
        self.base_lrs = state_dict["base_lrs"]
        self.last_epoch = state_dict["last_epoch"]
