from typing import Any, Callable
from functools import wraps


_grad_mode = False


def _set_grad_mode(mode: bool) -> None:
    global _grad_mode
    _grad_mode = mode


def is_grad_enabled() -> bool:
    return _grad_mode


class no_grad:
    def __init__(self) -> None:
        self.prev_state: bool = None

    def __enter__(self):
        self.prev_state = _grad_mode
        _set_grad_mode(False)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        _set_grad_mode(self.prev_state)

    def __call__(self, func: Callable[[Any], Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


class enable_grad:
    def __init__(self) -> None:
        self.prev_state: bool = None

    def __enter__(self):
        self.prev_state = _grad_mode
        _set_grad_mode(True)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        _set_grad_mode(self.prev_state)

    def __call__(self, func: Callable[[Any], Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper
