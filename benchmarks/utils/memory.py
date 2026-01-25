from __future__ import annotations
import gc
import tracemalloc


class MemoryProfiler:
    """Advanced memory profiler with snapshot comparison capabilities.

    This class provides detailed memory profiling by taking snapshots at different
    points during execution and comparing them to identify memory growth patterns.
    Useful for debugging memory leaks and understanding memory consumption over time.

    Attributes:
        snapshots (dict[str, tracemalloc.Snapshot]): Dictionary mapping snapshot
            names to their corresponding tracemalloc snapshots.
        is_running (bool): Whether the profiler is currently active.

    Examples:
        Basic profiling workflow:

        >>> profiler = MemoryProfiler()
        >>> profiler.start()
        >>> model = create_model()
        >>> profiler.snapshot("after_model")
        >>> train_data = load_dataset()
        >>> profiler.snapshot("after_data")
        >>> profiler.stop()
        >>> profiler.print_diff("after_model", "after_data")

        Tracking memory during training loop:

        >>> profiler = MemoryProfiler()
        >>> profiler.start()
        >>> for epoch in range(5):
        ...     train_epoch()
        ...     profiler.snapshot(f"epoch_{epoch}")
        >>> profiler.stop()
        >>> profiler.print_diff("epoch_0", "epoch_4", limit=5)

        Getting total allocated memory:

        >>> profiler = MemoryProfiler()
        >>> profiler.start()
        >>> process_data()
        >>> profiler.snapshot("checkpoint")
        >>> total_mb = profiler.get_total_allocated("checkpoint")
        >>> print(f"Total allocated: {total_mb:.2f} MB")

        Monitoring current memory during execution:

        >>> profiler = MemoryProfiler()
        >>> profiler.start()
        >>> for i in range(10):
        ...     do_work()
        ...     current, peak = profiler.get_current_memory()
        ...     print(f"Step {i}: {current:.2f} MB (peak: {peak:.2f} MB)")
    """

    snapshots: dict[str, tracemalloc.Snapshot]
    is_running: bool

    def __init__(self):
        """Initialize the memory profiler.

        Creates an empty profiler ready to start tracking memory allocations.

        Examples:
            >>> profiler = MemoryProfiler()
            >>> profiler.start()
        """
        self.snapshots = {}
        self.is_running = False

    def start(self):
        """Start memory profiling.

        Performs garbage collection for a clean baseline, starts tracemalloc,
        and takes an initial snapshot named 'start'.

        Examples:
            >>> profiler = MemoryProfiler()
            >>> profiler.start()
            >>> # Now profiler is tracking memory
        """
        gc.collect()
        tracemalloc.start()
        self.is_running = True
        self.snapshots["start"] = tracemalloc.take_snapshot()

    def snapshot(self, name: str):
        """Take a memory snapshot with the given name.

        Args:
            name: Unique identifier for this snapshot. Used later for comparisons.

        Raises:
            RuntimeError: If profiler hasn't been started yet.

        Examples:
            >>> profiler = MemoryProfiler()
            >>> profiler.start()
            >>> load_model()
            >>> profiler.snapshot("model_loaded")
            >>> process_data()
            >>> profiler.snapshot("data_processed")
        """
        if not self.is_running:
            raise RuntimeError("Profiler not started. Call start() first.")

        self.snapshots[name] = tracemalloc.take_snapshot()

    def stop(self):
        """Stop memory profiling.

        Takes a final snapshot named 'end' and stops tracemalloc. After calling
        this method, no more snapshots can be taken until start() is called again.

        Examples:
            >>> profiler = MemoryProfiler()
            >>> profiler.start()
            >>> do_work()
            >>> profiler.stop()
            >>> # Profiler is now stopped
        """
        if self.is_running:
            self.snapshots["end"] = tracemalloc.take_snapshot()
            tracemalloc.stop()
            self.is_running = False

    def get_current_memory(self) -> tuple[float, float]:
        """Get current and peak memory usage.

        Returns:
            tuple[float, float]: A tuple of (current_mb, peak_mb). Returns
                (0.0, 0.0) if profiler is not running.

        Examples:
            >>> profiler = MemoryProfiler()
            >>> profiler.start()
            >>> data = [0] * 1000000
            >>> current, peak = profiler.get_current_memory()
            >>> print(f"Current: {current:.2f} MB, Peak: {peak:.2f} MB")
            Current: 7.63 MB, Peak: 7.63 MB
        """
        if not self.is_running:
            return 0.0, 0.0

        current, peak = tracemalloc.get_traced_memory()
        return current / 1024**2, peak / 1024**2

    def print_diff(self, snapshot1: str, snapshot2: str, limit: int = 10):
        """Print memory differences between two snapshots.

        Args:
            snapshot1: Name of the first (earlier) snapshot.
            snapshot2: Name of the second (later) snapshot.
            limit: Maximum number of top differences to display. Defaults to 10.

        Raises:
            ValueError: If either snapshot name doesn't exist.

        Examples:
            >>> profiler = MemoryProfiler()
            >>> profiler.start()
            >>> profiler.snapshot("before")
            >>> data = process_large_file()
            >>> profiler.snapshot("after")
            >>> profiler.print_diff("before", "after", limit=5)
            ============================================================
            Memory diff: before -> after
            ============================================================
            Top 5 differences:
            /path/to/file.py:42: size=15.2 MB (+15.2 MB), count=1000 (+1000)
            ...
        """
        if snapshot1 not in self.snapshots or snapshot2 not in self.snapshots:
            raise ValueError(f"Snapshot not found: {snapshot1} or {snapshot2}")

        snap1 = self.snapshots[snapshot1]
        snap2 = self.snapshots[snapshot2]

        top_stats = snap2.compare_to(snap1, "lineno")

        print(f"\n{'='*60}")
        print(f"Memory diff: {snapshot1} -> {snapshot2}")
        print(f"{'='*60}")
        print(f"Top {limit} differences:")

        for stat in top_stats[:limit]:
            print(f"{stat}")

        print(f"{'='*60}\n")

    def get_total_allocated(self, snapshot_name: str) -> float:
        """Get total memory allocated at a specific snapshot.

        Args:
            snapshot_name: Name of the snapshot to analyze.

        Returns:
            float: Total allocated memory in megabytes.

        Raises:
            ValueError: If snapshot name doesn't exist.

        Examples:
            >>> profiler = MemoryProfiler()
            >>> profiler.start()
            >>> load_dataset()
            >>> profiler.snapshot("data_loaded")
            >>> total = profiler.get_total_allocated("data_loaded")
            >>> print(f"Total allocated: {total:.2f} MB")
            Total allocated: 256.34 MB
        """
        if snapshot_name not in self.snapshots:
            raise ValueError(f"Snapshot not found: {snapshot_name}")

        snapshot = self.snapshots[snapshot_name]
        total = sum(stat.size for stat in snapshot.statistics("filename"))
        return total / 1024**2
