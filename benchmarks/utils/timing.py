import time


class Timer:
    """Context manager for precise timing measurements.

    Examples:
        >>> with Timer() as t:
        ...     expensive_operation()
        >>> print(f"Took {t.elapsed:.3f}s")
    """

    def __init__(self):
        self.start_time: float = None
        self.elapsed: float = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
