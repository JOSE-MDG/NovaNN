from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Optional, TypeVar
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

TOptimizer = TypeVar("TOptimizer", bound="Optimizer")

class Optimizer:
    param_groups: ParamGroups
    state: State
    defaults: Defaults
    _step_pre_hook: list[StepHook]
    _step_post_hook: list[StepHook]

    def __init__(self, params: Iterable[Parameter], defaults: Defaults) -> None: ...
    def add_param_group(self, group: Group) -> None: ...
    def register_step_prev_hook(self, hook: StepHook) -> HooksHandle: ...
    def register_step_post_hook(self, hook: StepHook) -> HooksHandle: ...
    def _step_impl(self, closure: Closure = None) -> Optional[float]: ...
    def step(self, closure: Closure = None) -> Optional[float]: ...
    def zero_grad(self, set_to_none: bool = True) -> None: ...
    def state_dict(self) -> OptimizerStateDict: ...
    def load_state_dict(self, state_dict: OptimizerStateDict) -> None: ...
    def __repr__(self) -> str: ...
