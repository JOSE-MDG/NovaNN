from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova._typing import Hook


class HooksHandle:
    def __init__(self, hooks_list: list[Hook], hook_func: Hook):
        self.hooks_list: list[Hook] = hooks_list
        self.hooks_func: Hook = hook_func
        self._removed: bool = False

    def remove(self):
        if not self._removed and len(self.hooks_list) > 0:
            self.hooks_list.remove(self.hooks_func)
