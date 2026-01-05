import threading
from typing import Any, Callable
from functools import wraps

# Thread-local storage to handle autograd state independently on each thread.
_storage = threading.local()


def _set_grad_mode(mode: bool) -> None:
    """
    Internal helper to enable or disable gradient tracking for the current thread.

    This function mutates the thread-local autograd state and should only be
    used internally by context managers such as `no_grad` or `enable_grad`.
    """
    _storage.grad_mode = mode


def is_grad_enabled() -> bool:
    """
    Check whether gradient tracking is currently enabled in the current thread.

    Returns:
        True if autograd is active and operations will be tracked.
        False if gradient computation is disabled.
        Defaults to True if the state hasn't been explicitly set in the thread.
    """
    if not hasattr(_storage, "grad_mode"):
        _storage.grad_mode = True
    return _storage.grad_mode


class no_grad:
    """
    Context manager and decorator that disables gradient tracking.

    While active, all operations in the current thread are executed without
    building the computation graph. This is useful for inference, evaluation,
    or parameter updates where memory efficiency is prioritized and
    backpropagation is not required.

    Note: This is thread-safe; disabling gradients in one thread will not
    affect gradient computation in other threads.

    Example (context manager):
        >>> import nova
        >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
        >>> with nova.no_grad():
        ...     y = x * 2
        ...     print(y.requires_grad)  # output: False
        ...     print(x.requires_grad)  # output: True (unaffected)
        >>>
        >>> # Outside the block
        >>> z = x * 2
        >>> print(z.requires_grad)
        True

    Example (decorator):
        >>> @no_grad()
        ... def inference(x):
        ...     return x * 2
    """

    def __init__(self) -> None:
        self.prev_state: bool = None

    def __enter__(self):
        self.prev_state = is_grad_enabled()
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
    """
    Context manager and decorator that explicitly enables gradient tracking.

    This is mainly useful to re-enable autograd inside a `no_grad` block,
    or to ensure gradients are tracked regardless of the outer context
    in the current thread.

    Example:
        >>> with no_grad():
        ...     with enable_grad():
        ...         y = x * 2
        >>> y.requires_grad
        True

    Example (decorator):
        >>> @enable_grad()
        ... def training_step(x):
        ...     return (x * 2).sum()
    """

    def __init__(self) -> None:
        self.prev_state: bool = None

    def __enter__(self):
        self.prev_state = is_grad_enabled()
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
