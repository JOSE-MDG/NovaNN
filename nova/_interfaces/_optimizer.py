from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Optional
from nova.utils import registry_class
from nova.autograd.utils.hooks import HooksHandle

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
    def __init__(self, params: Iterable[Parameter], defaults: Defaults) -> None:
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
        self._step_pre_hook.append(hook)
        handle = HooksHandle(self._step_pre_hook, hook)
        return handle

    def register_step_post_hook(self, hook: StepHook) -> HooksHandle:
        self._step_post_hook.append(hook)
        handle = HooksHandle(self._step_post_hook, hook)
        return handle

    def _step_impl(self, closure: Closure = None) -> Optional[float]:
        raise NotImplementedError

    def step(self, closure: Closure = None) -> Optional[float]:

        for hook in self._step_pre_hook:
            hook(self)

        loss = self._step_impl(closure)

        for hook in self._step_post_hook:
            hook(self)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                param.zero_grad(set_to_none=set_to_none)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " ("

        for i, group in enumerate(self.param_groups):
            format_string += f"\nParameter Group {i}"

            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"\n    {key}: {group[key]}"

        format_string += "\n)"
        return format_string

    def state_dict(self) -> OptimizerStateDict:

        return {
            "state": self.state,
            "param_groups": [
                {k: v for k, v in group.items() if k != "params"}
                for group in self.param_groups
            ],
        }

    def load_state_dict(self, state_dict: OptimizerStateDict) -> None:

        self.state = state_dict["state"]

        for i, group in enumerate(state_dict["param_groups"]):
            self.param_groups[i].update(group["state"])
