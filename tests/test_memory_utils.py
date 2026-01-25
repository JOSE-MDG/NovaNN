import pytest
import gc
from nova.utils.memory import (
    MemoryTracker,
    quick_memory_check,
    compare_memory,
)


class TestMemoryTracker:
    """Test the MemoryTracker context manager."""

    def test_basic_context_manager(self):
        """Test basic usage as context manager."""
        with MemoryTracker() as mem:
            # Allocate some memory
            [i for i in range(100000)]

        # Should have tracked memory
        assert mem.peak > 0
        assert mem.current > 0
        assert mem.baseline >= 0

    def test_memory_properties(self):
        """Test memory property accessors."""
        with MemoryTracker() as mem:
            [0] * 1000000

        # Test all properties return valid values
        assert mem.peak_mb > 0
        assert mem.current_mb > 0
        assert mem.peak_kb > 0
        assert mem.current_kb > 0


class TestQuickMemoryCheck:
    """Test the quick_memory_check function."""

    def test_basic_function_profiling(self):
        """Test profiling a simple function."""

        def create_list(n):
            return [i for i in range(n)]

        stats = quick_memory_check(create_list, 10000)

        # Check result is correct
        assert len(stats["result"]) == 10000

        # Memory should be tracked
        assert stats["peak_mb"] > 0

    def test_with_kwargs(self):
        """Test profiling with keyword arguments."""

        def multiply_list(data, factor=1):
            return [x * factor for x in data]

        stats = quick_memory_check(multiply_list, [1, 2, 3], factor=10)

        # Verify result
        assert stats["result"] == [10, 20, 30]

        # Memory tracked
        assert stats["peak_kb"] > 0


class TestCompareMemory:
    """Test the compare_memory function."""

    def test_memory_comparison(self):
        """Test comparing memory between two functions."""

        def func_small():
            return [i for i in range(1000)]

        def func_large():
            return [i for i in range(100000)]

        small_peak, large_peak, ratio = compare_memory(
            func_small, func_large, verbose=False
        )

        assert large_peak > small_peak
        assert ratio <= 1.0

        assert small_peak > 0
        assert large_peak > 0


class TestMemoryTrackerAdvanced:
    """Test advanced MemoryTracker features."""

    def test_get_top_stats(self):
        """Test getting top memory allocations."""
        with MemoryTracker() as mem:
            [i for i in range(50000)]
            [i * 2 for i in range(50000)]

        # Get top stats
        top_stats = mem.get_top_stats(5)

        # Should return a list
        assert isinstance(top_stats, list)

        # Should have at most 5 items
        assert len(top_stats) <= 5

    def test_verbose_mode(self, capsys):
        """Test verbose mode prints statistics."""
        with MemoryTracker(verbose=True):
            [i for i in range(10000)]

        # Capture printed output
        captured = capsys.readouterr()

        # Should have printed something
        assert "Memory Usage Statistics" in captured.out
        assert "Peak memory:" in captured.out
        assert "Current memory:" in captured.out


class TestMemoryContextBehavior:
    """Test MemoryTracker context manager behavior."""

    def test_multiple_sequential_uses(self):
        """Test using MemoryTracker multiple times sequentially."""
        peaks = []

        for _ in range(3):
            gc.collect()
            with MemoryTracker() as mem:
                [i for i in range(10000)]
                peaks.append(mem.peak_mb)

        # All runs should track memory
        assert all([p >= 0 for p in peaks])
