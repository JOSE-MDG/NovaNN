from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova._typing import Hooks, HooksList


class HooksHandle:
    def __init__(
        self,
        hooks_list: HooksList,
        hook_func: Hooks,
    ):
        self.hooks_list: HooksList = hooks_list
        self.hooks_func: Hooks = hook_func
        self._removed: bool = False

    def remove(self) -> None:
        if not self._removed and len(self.hooks_list) > 0:
            self.hooks_list.remove(self.hooks_func)
