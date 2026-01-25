from __future__ import annotations
import gc
import tracemalloc
from typing import Optional, TypeVar, Callable, Any

_T = TypeVar("_T")


class MemoryTracker:
    """Context manager for tracking memory usage during code execution.

    This class provides a simple way to monitor memory consumption using Python's
    tracemalloc module. It automatically handles garbage collection for accurate
    baseline measurements and tracks both peak and current memory usage.

    Attributes:
        verbose (bool): Whether to print memory statistics on exit.
        baseline (int): Baseline memory usage in bytes.
        peak (int): Peak memory usage in bytes (baseline-adjusted).
        current (int): Current memory usage in bytes (baseline-adjusted).

    Examples:
        Basic usage with context manager:

        >>> with MemoryTracker() as mem:
        ...     data = [i for i in range(1000000)]
        >>> print(f"Peak: {mem.peak_mb:.2f} MB")
        Peak: 38.15 MB

        Verbose mode for automatic reporting:

        >>> with MemoryTracker(verbose=True) as mem:
        ...     model = create_large_model()
        ==================================================
        Memory Usage Statistics
        ==================================================
        Peak memory:         125.43 MB
        Current memory:       98.21 MB
        ==================================================

        Analyzing top allocations:

        >>> with MemoryTracker() as mem:
        ...     data = process_large_dataset()
        >>> top_stats = mem.get_top_stats(5)
        >>> for stat in top_stats:
        ...     print(stat)

        Getting memory in different units:

        >>> with MemoryTracker() as mem:
        ...     result = compute_something()
        >>> print(f"Peak: {mem.peak_mb:.2f} MB")
        >>> print(f"Peak: {mem.peak_kb:.2f} KB")
    """

    verbose: bool
    baseline: int
    peak: int
    current: int
    _snapshot_start: Optional[tracemalloc.Snapshot]
    _snapshot_end: Optional[tracemalloc.Snapshot]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.baseline = 0
        self.peak = 0
        self.current = 0
        self._snapshot_start = None
        self._snapshot_end = None

    def __enter__(self):
        """Start tracking memory.

        Performs garbage collection to establish a clean baseline, then begins
        memory tracking using tracemalloc.

        Returns:
            MemoryTracker: Self reference for context manager usage.
        """
        # Force garbage collection to get clean baseline
        gc.collect()

        # Start tracking
        tracemalloc.start()
        self._snapshot_start = tracemalloc.take_snapshot()
        self.baseline = tracemalloc.get_traced_memory()[0]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking and compute statistics.

        Captures final memory measurements, stops tracemalloc, adjusts values
        for baseline, and optionally prints statistics.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.

        Returns:
            bool: Always returns False to propagate exceptions.
        """
        # Get current and peak memory
        self.current, self.peak = tracemalloc.get_traced_memory()

        # Take final snapshot
        self._snapshot_end = tracemalloc.take_snapshot()

        # Stop tracking
        tracemalloc.stop()

        # Adjust for baseline
        self.peak -= self.baseline
        self.current -= self.baseline

        if self.verbose:
            self._print_stats()

        return False

    def _print_stats(self):
        """Print formatted memory statistics to stdout.

        Internal method that displays peak and current memory usage in a
        formatted table.
        """
        print(f"\n{'='*50}")
        print("Memory Usage Statistics")
        print(f"{'='*50}")
        print(f"Peak memory:    {self.peak / 1024**2:>10.2f} MB")
        print(f"Current memory: {self.current / 1024**2:>10.2f} MB")
        print(f"{'='*50}\n")

    def get_top_stats(self, limit: int = 10):
        """Get top memory allocations ordered by size.

        Args:
            limit: Maximum number of allocations to return. Defaults to 10.

        Returns:
            list: List of StatisticDiff objects representing the top memory
                allocations, or empty list if no end snapshot exists.

        Examples:
            >>> with MemoryTracker() as mem:
            ...     data = [i**2 for i in range(100000)]
            >>> stats = mem.get_top_stats(3)
            >>> for stat in stats:
            ...     print(f"{stat.filename}:{stat.lineno}: {stat.size_diff / 1024:.1f} KB")
        """
        if self._snapshot_end is None:
            return []

        top_stats = self._snapshot_end.compare_to(self._snapshot_start, "lineno")

        return top_stats[:limit]

    @property
    def peak_mb(self) -> float:
        """Peak memory usage in megabytes.

        Returns:
            float: Peak memory in MB.

        Examples:
            >>> with MemoryTracker() as mem:
            ...     data = [0] * 1000000
            >>> print(f"{mem.peak_mb:.2f} MB")
            7.63 MB
        """
        return self.peak / 1024**2

    @property
    def current_mb(self) -> float:
        """Current memory usage in megabytes.

        Returns:
            float: Current memory in MB.

        Examples:
            >>> with MemoryTracker() as mem:
            ...     data = [0] * 1000000
            >>> print(f"{mem.current_mb:.2f} MB")
            7.63 MB
        """
        return self.current / 1024**2

    @property
    def peak_kb(self) -> float:
        """Peak memory usage in kilobytes.

        Returns:
            float: Peak memory in KB.

        Examples:
            >>> with MemoryTracker() as mem:
            ...     data = [0] * 100000
            >>> print(f"{mem.peak_kb:.2f} KB")
            781.25 KB
        """
        return self.peak / 1024

    @property
    def current_kb(self) -> float:
        """Current memory usage in kilobytes.

        Returns:
            float: Current memory in KB.

        Examples:
            >>> with MemoryTracker() as mem:
            ...     data = [0] * 100000
            >>> print(f"{mem.current_kb:.2f} KB")
            781.25 KB
        """
        return self.current / 1024


def quick_memory_check(func: Callable[[_T], _T], *args, **kwargs) -> dict:
    """Convenience function for quick memory profiling of a function call.

    Executes the given function with the provided arguments while tracking
    memory usage. Returns both the function result and memory statistics.

    Args:
        func: The function to profile.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        dict: Dictionary containing:
            - peak_mb (float): Peak memory usage in megabytes
            - current_mb (float): Current memory usage in megabytes
            - peak_kb (float): Peak memory usage in kilobytes
            - current_kb (float): Current memory usage in kilobytes
            - result: The return value of the function

    Examples:
        Simple function profiling:

        >>> def create_list(n):
        ...     return [i for i in range(n)]
        >>> stats = quick_memory_check(create_list, 1000000)
        >>> print(f"Peak: {stats['peak_mb']:.2f} MB")
        >>> print(f"Result length: {len(stats['result'])}")
        Peak: 38.15 MB
        Result length: 1000000

        Profiling with keyword arguments:

        >>> def process_data(data, multiply=1):
        ...     return [x * multiply for x in data]
        >>> stats = quick_memory_check(process_data, [1,2,3], multiply=10)
        >>> print(f"Memory used: {stats['current_kb']:.2f} KB")
        >>> print(f"Result: {stats['result']}")

        Using in a loop to compare approaches:

        >>> approaches = [approach_a, approach_b, approach_c]
        >>> for i, func in enumerate(approaches):
        ...     stats = quick_memory_check(func, large_dataset)
        ...     print(f"Approach {i}: {stats['peak_mb']:.2f} MB")
    """
    gc.collect()

    with MemoryTracker() as mem:
        result = func(*args, **kwargs)

    return {
        "peak_mb": mem.peak_mb,
        "current_mb": mem.current_mb,
        "peak_kb": mem.peak_kb,
        "current_kb": mem.current_kb,
        "result": result,
    }


def compare_memory(
    input_func: Callable[[_T], _T],
    other_func: Callable[[_T], _T],
    *args: Any,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[float, float, float]:
    """
    Compare memory usage between implementations.

    Examples:
        >>> from nova.utils.memory import compare_memory
        >>> nova_peak, torch_peak, ratio = compare_memory(
        ...    nova_forward,
        ...    torch_forward,
        ...    x_nova, x_torch
        ... )

    Args:
        nova_func: NovaNN function to benchmark
        torch_func: PyTorch function to benchmark
        *args: Arguments to pass to functions
        verbose: If True, print comparison
        **kwargs: Keyword arguments to pass to functions

    Returns:
        Tuple of (nova_peak_mb, torch_peak_mb, ratio)
    """
    # Measure NovaNN
    gc.collect()
    with MemoryTracker() as nova_mem:
        input_func(*args, **kwargs)

    # Measure PyTorch
    gc.collect()
    with MemoryTracker() as torch_mem:
        other_func(*args, **kwargs)

    ratio = nova_mem.peak_mb / torch_mem.peak_mb if torch_mem.peak_mb > 0 else 0

    if verbose:
        print(f"\n{'='*50}")
        print("Memory Comparison: NovaNN vs PyTorch")
        print(f"{'='*50}")
        print(f"NovaNN peak:  {nova_mem.peak_mb:>10.2f} MB")
        print(f"PyTorch peak: {torch_mem.peak_mb:>10.2f} MB")
        print(f"Ratio:        {ratio:>10.2f}x")
        print(f"{'='*50}\n")

    return nova_mem.peak_mb, torch_mem.peak_mb, ratio
