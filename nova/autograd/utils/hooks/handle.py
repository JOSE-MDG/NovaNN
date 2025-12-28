from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova._typing import Hook, StepHook


class HooksHandle:
    def __init__(
        self,
        hooks_list: list[Hook] | list[StepHook] | list[Hook | StepHook],
        hook_func: Hook | StepHook,
    ):
        self.hooks_list: list[Hook] | list[StepHook] | list[Hook | StepHook] = (
            hooks_list
        )
        self.hooks_func: Hook | StepHook = hook_func
        self._removed: bool = False

    def remove(self) -> None:
        if not self._removed and len(self.hooks_list) > 0:
            self.hooks_list.remove(self.hooks_func)  # type: ignore
