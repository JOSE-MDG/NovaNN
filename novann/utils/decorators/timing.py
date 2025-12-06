import time
from functools import wraps
from typing import Callable, Any
from novann.utils.log_config import logger


def chronometer(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    Returns the original function result unchanged.

    Args:
        func: Function to be timed.

    Returns:
        Wrapped function that logs execution time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Start timing
        start = time.perf_counter()

        # Execute function and capture result
        result = func(*args, **kwargs)

        # Calculate elapsed time
        elapsed = time.perf_counter() - start

        # Smart formatting with extended time ranges
        if elapsed < 1e-6:  # < 1 microsecond
            time_str = f"{elapsed * 1e9:.0f}ns"
        elif elapsed < 1e-3:  # < 1 millisecond
            time_str = f"{elapsed * 1e6:.0f}μs"
        elif elapsed < 1:  # < 1 second
            time_str = f"{elapsed * 1e3:.0f}ms"
        elif elapsed < 60:  # < 1 minute
            time_str = f"{elapsed:.2f}s"
        elif elapsed < 3600:  # < 1 hour
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.1f}s" if seconds >= 1 else f"{minutes}m"
        else:  # ≥ 1 hour
            hours = int(elapsed // 3600)
            remaining_minutes = int((elapsed % 3600) // 60)
            remaining_seconds = elapsed % 60

            if remaining_minutes == 0 and remaining_seconds < 1:
                time_str = f"{hours}h"
            elif remaining_minutes > 0 and remaining_seconds < 1:
                time_str = f"{hours}h {remaining_minutes}m"
            else:
                time_str = f"{hours}h {remaining_minutes}m {remaining_seconds:.0f}s"

        # Log with appropriate emoji
        emoji = "⚡" if elapsed < 1 else "⏱️" if elapsed < 60 else "🐢"
        logger.info(f"{emoji} {func.__name__}: {time_str}")

        # Return original result unchanged
        return result

    return wrapper
