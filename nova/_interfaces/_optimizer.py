from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Optional
from nova.utils import registry_class
from nova.utils.hooks import HooksHandle

if TYPE_CHECKING:
    from nova.nn import Parameter
    from nova._typing import (
        StepHook,
        Closure,
        ParamGroups,
        State,
        Defaults,
        OptimizerStateDict,
        Group,
    )


@registry_class
class Optimizer:
    """
    Base class for all optimizers.

    An Optimizer manages parameter groups, optimizer state, and the
    optimization step logic. Subclasses must implement `_step_impl`.

    This class also supports pre- and post-step hooks, useful for
    logging, gradient clipping, or custom behaviors.
    """

    def __init__(self, params: Iterable[Parameter], defaults: Defaults) -> None:
        """
        Initialize the optimizer.

        Args:
            params: Iterable of parameters or parameter groups.
                Can be:
                - an iterable of Parameters
                - an iterable of dicts with a "params" key
            defaults: Dictionary of default hyperparameters (e.g., lr).

        Raises:
            ValueError: If `params` is empty.
        """

        self.param_groups: ParamGroups = []
        self.state: State = {}
        self.defaults: Defaults = defaults
        self._step_pre_hook: list[StepHook] = []
        self._step_post_hook: list[StepHook] = []

        params = list(params)

        if len(params) == 0:
            raise ValueError("Optimizer got an empty parameter list")

        if isinstance(params[0], dict):
            param_group = params
        else:
            param_group = [{"params": params}]

        for group in param_group:
            self.add_param_group(group=group)

    def add_param_group(self, group: Group) -> None:
        """
        Add a parameter group to the optimizer.

        Parameter groups allow different hyperparameters for different
        subsets of parameters.

        Args:
            group: Dictionary containing a "params" key and optional
                hyperparameters.

        Raises:
            KeyError: If "params" key is missing.
            ValueError: If "params" is empty.
        """

        if "params" not in group:
            raise KeyError("param_group must have a 'params' key")

        params = list(group["params"])

        if len(params) == 0:
            raise ValueError("param_group 'params' is empty")

        for name, value in self.defaults.items():
            group.setdefault(name, value)

        group["params"] = params

        self.param_groups.append(group)

    def register_step_prev_hook(self, hook: StepHook) -> HooksHandle:
        """
        Register a hook to be called before each optimization step.

        Args:
            hook: Callable receiving the optimizer instance.

        Returns:
            A handle that can be used to remove the hook.
        """
        self._step_pre_hook.append(hook)
        handle = HooksHandle(self._step_pre_hook, hook)
        return handle

    def register_step_post_hook(self, hook: StepHook) -> HooksHandle:
        """
        Register a hook to be called after each optimization step.

        Args:
            hook: Callable receiving the optimizer instance.

        Returns:
            A handle that can be used to remove the hook.
        """
        self._step_post_hook.append(hook)
        handle = HooksHandle(self._step_post_hook, hook)
        return handle

    def _step_impl(self, closure: Closure = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Subclasses must override this method to implement the update rule.

        Args:
            closure: Optional callable that reevaluates the model and
                returns the loss.

        Returns:
            The loss value if provided by the closure.
        """
        raise NotImplementedError

    def step(self, closure: Closure = None) -> Optional[float]:
        """
        Perform an optimization step.

        Executes registered pre-step hooks, calls the internal step
        implementation, and then executes post-step hooks.

        Args:
            closure: Optional callable returning the loss.

        Returns:
            The loss value if provided.

        Example:
            >>> optimizer = nova.optim.SGD(model.parameters(), lr=0.1)
            >>> for epoch in range(100):
            ...     optimizer.zero_grad()
            ...     loss = criterion(model(x), y)
            ...     loss.backward()
            ...     optimizer.step()
        """

        for hook in self._step_pre_hook:
            hook(self)

        loss = self._step_impl(closure)

        for hook in self._step_post_hook:
            hook(self)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Clear gradients of all optimized parameters.

        Args:
            set_to_none: If True, gradients are set to None instead of zero.
        """
        for group in self.param_groups:
            for param in group["params"]:
                param.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> OptimizerStateDict:
        """
        Return the optimizer state.

        Returns:
            A dictionary containing optimizer state and parameter group
            hyperparameters.
        """

        return {
            "state": self.state,
            "param_groups": [
                {k: v for k, v in group.items() if k != "params"}
                for group in self.param_groups
            ],
        }

    def load_state_dict(self, state_dict: OptimizerStateDict) -> None:
        """
        Load the optimizer state.

        Args:
            state_dict: Optimizer state dictionary obtained from `state_dict`.
        """

        self.state = state_dict["state"]

        for i, group in enumerate(state_dict["param_groups"]):
            self.param_groups[i].update(group)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " ("

        for i, group in enumerate(self.param_groups):
            format_string += f"\nParameter Group {i}"

            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"\n    {key}: {group[key]}"

        format_string += "\n)"
        return format_string
