import time
import numpy as np
from functools import wraps
from typing import TypeVar, Callable, Any, Optional
from nova.utils.logger import logger

_T = TypeVar("_T")


def benchmark(
    func: Callable[[_T], _T],
    *args: Any,
    n_iters: int = 100,
    warmup: int = 10,
    **kwargs: Any,
) -> tuple[Any, float, float]:
    """
    Benchmark a function with multiple iterations.

    Args:
        func: Function to benchmark.
        *args: Positional arguments for the function.
        n_iters: Number of timed iterations.
        warmup: Number of warmup iterations.
        **kwargs: Keyword arguments for the function.

    Returns:
        Tuple of (result, mean_time, std_time)

    Examples:
        >>> result, mean, std = benchmark(matrix_multiply, A, B, n_iters=100)
        >>> print(f"Average: {mean*1000:.2f}ms ± {std*1000:.2f}ms")
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    # Benchmark
    times = []
    result = None

    for _ in range(n_iters):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = np.mean(times).item()
    std_time = np.std(times).item()

    return result, mean_time, std_time


def chronometer(
    func: Optional[Callable[[_T], _T]] = None,
    *,
    n_iters: int = 1,
    warmup: int = 0,
    return_time: bool = False,
    verbose: bool = True,
) -> Callable[[_T], _T]:
    """
    Decorator to measure and log function execution time with benchmarking support.

    Args:
        func: Function to be timed.
        n_iters: Number of iterations to run (for averaging).
        warmup: Number of warmup iterations (not counted in timing).
        return_time: If True, returns (result, avg_time) instead of just result.
        verbose: If True, logs execution time.

    Returns:
        Wrapped function that optionally returns timing information.

    Examples:
        >>> @chronometer
        ... def my_func():
        ...     time.sleep(0.1)

        >>> @chronometer(n_iters=100, warmup=10)
        ... def benchmark_func():
        ...     return x @ y

        >>> @chronometer(n_iters=50, return_time=True, verbose=False)
        ... def timed_func():
        ...     return compute()
        >>> result, elapsed = timed_func()
    """

    def decorator(fn: Callable[[_T], _T]) -> Callable[[_T], _T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            # Warmup iterations
            for _ in range(warmup):
                _ = fn(*args, **kwargs)

            # Timed iterations
            times = []
            result = None

            for _ in range(n_iters):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            # Calculate average time
            avg_time = sum(times) / len(times)

            # Smart formatting
            if verbose:
                time_str = _format_time(avg_time)
                emoji = "⚡" if avg_time < 1 else "⏱️" if avg_time < 60 else "🐢"

                if n_iters > 1:
                    logger.info(
                        f"{emoji} {fn.__name__}: {time_str} (avg over {n_iters} runs)"
                    )
                else:
                    logger.info(f"{emoji} {fn.__name__}: {time_str}")

            # Return based on return_time flag
            if return_time:
                return result, avg_time
            return result

        return wrapper

    # Handle both @chronometer and @chronometer(...) syntax
    if func is None:
        # Called with arguments: @chronometer(n_iters=100)
        return decorator
    else:
        # Called without arguments: @chronometer
        return decorator(func)


def _format_time(elapsed: float) -> str:
    """Format elapsed time with appropriate units."""
    if elapsed < 1e-6:  # < 1 microsecond
        return f"{elapsed * 1e9:.0f}ns"
    elif elapsed < 1e-3:  # < 1 millisecond
        return f"{elapsed * 1e6:.0f}μs"
    elif elapsed < 1:  # < 1 second
        return f"{elapsed * 1e3:.2f}ms"
    elif elapsed < 60:  # < 1 minute
        return f"{elapsed:.2f}s"
    elif elapsed < 3600:  # < 1 hour
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.1f}s" if seconds >= 1 else f"{minutes}m"
    else:  # ≥ 1 hour
        hours = int(elapsed // 3600)
        remaining_minutes = int((elapsed % 3600) // 60)
        remaining_seconds = elapsed % 60

        if remaining_minutes == 0 and remaining_seconds < 1:
            return f"{hours}h"
        elif remaining_minutes > 0 and remaining_seconds < 1:
            return f"{hours}h {remaining_minutes}m"
        else:
            return f"{hours}h {remaining_minutes}m {remaining_seconds:.0f}s"
