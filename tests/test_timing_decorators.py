import time
import pytest
from nova.utils import benchmark
from nova.utils.decorators import chronometer


class TestBenchmark:
    """Test the benchmark function."""

    def test_basic_benchmark(self):
        """Test basic benchmarking of a function."""

        def simple_func(n):
            return sum(range(n))

        result, mean_time, std_time = benchmark(simple_func, 1000, n_iters=10, warmup=2)

        # Check result is correct
        assert result == sum(range(1000))

        # Check timing metrics exist and are positive
        assert mean_time > 0
        assert std_time >= 0

    def test_benchmark_with_kwargs(self):
        """Test benchmark with keyword arguments."""

        def compute(x, multiplier=1):
            return x * multiplier

        result, mean_time, _ = benchmark(compute, 5, multiplier=10, n_iters=5, warmup=1)

        assert result == 50
        assert mean_time > 0

    def test_benchmark_warmup_effect(self):
        """Test that warmup iterations don't affect timing."""
        counter = []

        def counting_func():
            counter.append(1)
            return len(counter)

        _, _, _ = benchmark(counting_func, n_iters=5, warmup=3)

        # Should have run warmup + n_iters times
        assert len(counter) == 8  # 3 warmup + 5 timed

    def test_benchmark_consistency(self):
        """Test that benchmark produces consistent results."""

        def stable_func():
            time.sleep(0.001)  # 1ms
            return 42

        _, mean_time, std_time = benchmark(stable_func, n_iters=5, warmup=1)

        # Mean should be around 1ms (with some tolerance)
        assert 0.0005 < mean_time < 0.005  # 0.5ms to 5ms

        # Std should be relatively small for stable function
        assert std_time < mean_time


class TestChronometer:
    """Test the chronometer decorator."""

    def test_basic_decorator(self):
        """Test basic decorator usage."""

        @chronometer
        def simple_func():
            time.sleep(0.001)
            return 42

        result = simple_func()

        # Check result
        assert result == 42

    def test_decorator_with_iterations(self):
        """Test decorator with multiple iterations."""

        @chronometer(n_iters=5, warmup=2, verbose=False)
        def test_func():
            return sum(range(100))

        result = test_func()
        assert result == sum(range(100))

    def test_return_time_flag(self):
        """Test decorator with return_time=True."""

        @chronometer(return_time=True, verbose=False)
        def timed_func():
            time.sleep(0.001)
            return "result"

        result, elapsed = timed_func()

        # Check result
        assert result == "result"

        # Check elapsed time exists and is reasonable
        assert isinstance(elapsed, float)
        assert elapsed > 0
        assert elapsed < 1.0  # Should be well under 1 second

    def test_verbose_false(self, capsys):
        """Test that verbose=False suppresses output."""

        @chronometer(verbose=False)
        def silent_func():
            return 123

        result = silent_func()

        assert result == 123

        # No output should be printed
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_warmup_iterations(self):
        """Test that warmup iterations are executed."""
        counter = []

        @chronometer(n_iters=3, warmup=2, verbose=False)
        def counting_func():
            counter.append(1)
            return len(counter)

        counting_func()

        # Should run warmup + n_iters
        assert len(counter) == 5  # 2 warmup + 3 timed

    def test_return_time_with_iterations(self):
        """Test return_time with multiple iterations."""

        @chronometer(n_iters=5, return_time=True, verbose=False)
        def avg_func():
            time.sleep(0.001)
            return 100

        result, avg_time = avg_func()

        assert result == 100
        assert avg_time > 0
        # Average of 5 runs with 1ms sleep each
        assert 0.0005 < avg_time < 0.01
