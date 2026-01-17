from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova._typing import Hooks, HooksList


class HooksHandle:
    """
    Handle for managing a registered hook in NovaNN.

    This object allows easy removal of hooks from a list once they are no longer needed.

    Attributes:
        hooks_list (HooksList): The list where the hook is registered.
        hooks_func (Hooks): The actual hook function.
        _removed (bool): Flag indicating whether the hook has been removed.

    Examples:
        >>> from nova.utils.hooks import HooksHandle
        >>> hooks_list = []
        >>> def my_hook(x): return x
        >>> handle = HooksHandle(hooks_list, my_hook)
        >>> hooks_list.append(my_hook)
        >>> handle.remove()
        >>> my_hook in hooks_list
        False
    """

    def __init__(
        self,
        hooks_list: HooksList,
        hook_func: Hooks,
    ):
        self.hooks_list: HooksList = hooks_list
        self.hooks_func: Hooks = hook_func
        self._removed: bool = False

    def remove(self) -> None:
        """Remove the hook from its list if it has not been removed yet."""
        if not self._removed and len(self.hooks_list) > 0:
            self.hooks_list.remove(self.hooks_func)
            self._removed = True
