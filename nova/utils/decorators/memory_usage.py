from functools import wraps
from nova.utils.memory import MemoryTracker
from typing import Optional, Callable, TypeVar

_T = TypeVar("_T")


def measure_memory(
    func: Optional[Callable[[_T], _T]] = None,
    *,
    verbose: bool = False,
    return_memory: bool = False,
) -> Callable[[_T], _T]:
    """
    Decorator to measure memory usage of a function.

    Usage:
        @measure_memory
        def my_function():
            # Your code here
            pass

        # Or with parameters
        @measure_memory(verbose=True, return_memory=True)
        def my_function():
            # Your code here
            pass

        result, (peak_mem, current_mem) = my_function()

    Args:
        func: Function to decorate (automatically provided)
        verbose: If True, print memory stats
        return_memory: If True, return (result, (peak_mb, current_mb))

    Returns:
        Decorated function
    """

    def decorator(f: Callable[[_T], _T]) -> Callable[[_T], tuple[_T, (int, int)]]:
        @wraps(f)
        def wrapper(*args, **kwargs) -> Callable[[_T], tuple[_T, (int, int)]]:
            with MemoryTracker(verbose=verbose) as mem:
                result = f(*args, **kwargs)

            if return_memory:
                return result, (mem.peak_mb, mem.current_mb)
            return result

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
