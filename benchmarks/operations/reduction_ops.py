"""
Benchmark: Reduction Operations (CPU)

Measures the performance of reduction operations (sum, mean, var, std, min, max)
on CPU across different tensor sizes.

Comparison: NovaNN vs PyTorch
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import nova
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nova.utils.decorators import chronometer
from utils.timing import Timer

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (4.5, 4.5),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    }
)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
OUTPUT_DIR = Path("images/benchmarks/operations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def nova_sum(x):
    """NovaNN sum reduction."""
    return x.sum()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_sum(x):
    """PyTorch sum reduction."""
    return x.sum()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def nova_mean(x):
    """NovaNN mean reduction."""
    return x.mean()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_mean(x):
    """PyTorch mean reduction."""
    return x.mean()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def nova_var(x):
    """NovaNN variance reduction."""
    return x.var()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_var(x):
    """PyTorch variance reduction."""
    return x.var()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def nova_std(x):
    """NovaNN standard deviation reduction."""
    return x.std()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_std(x):
    """PyTorch standard deviation reduction."""
    return x.std()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def nova_min(x):
    """NovaNN min reduction."""
    return x.min()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_min(x):
    """PyTorch min reduction."""
    return x.min()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def nova_max(x):
    """NovaNN max reduction."""
    return x.max()


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_max(x):
    """PyTorch max reduction."""
    return x.max()


def benchmark_basic_reductions():
    """Benchmark basic reduction operations (sum, mean, min, max)."""
    sizes = [100, 1000, 10000, 100000, 1000000]

    nova_sum_times = []
    torch_sum_times = []
    nova_mean_times = []
    torch_mean_times = []
    nova_min_times = []
    torch_min_times = []
    nova_max_times = []
    torch_max_times = []

    for size in sizes:
        print(f"\nSize: {size:,}")

        # Create tensors
        x_nova = nova.randn(size)
        x_torch = torch.randn(size)

        # Sum
        _, t_nova = nova_sum(x_nova)
        nova_sum_times.append(t_nova * 1000)

        _, t_torch = torch_sum(x_torch)
        torch_sum_times.append(t_torch * 1000)

        print(f"  Sum:  NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms")

        # Mean
        _, t_nova = nova_mean(x_nova)
        nova_mean_times.append(t_nova * 1000)

        _, t_torch = torch_mean(x_torch)
        torch_mean_times.append(t_torch * 1000)

        print(f"  Mean: NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms")

        # Min
        _, t_nova = nova_min(x_nova)
        nova_min_times.append(t_nova * 1000)

        _, t_torch = torch_min(x_torch)
        torch_min_times.append(t_torch * 1000)

        print(f"  Min:  NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms")

        # Max
        _, t_nova = nova_max(x_nova)
        nova_max_times.append(t_nova * 1000)

        _, t_torch = torch_max(x_torch)
        torch_max_times.append(t_torch * 1000)

        print(f"  Max:  NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms")

    return {
        "sizes": sizes,
        "nova_sum": nova_sum_times,
        "torch_sum": torch_sum_times,
        "nova_mean": nova_mean_times,
        "torch_mean": torch_mean_times,
        "nova_min": nova_min_times,
        "torch_min": torch_min_times,
        "nova_max": nova_max_times,
        "torch_max": torch_max_times,
    }


def benchmark_statistical_reductions():
    """Benchmark statistical reduction operations (var, std)."""
    sizes = [100, 1000, 10000, 100000, 1000000]

    nova_var_times = []
    torch_var_times = []
    nova_std_times = []
    torch_std_times = []

    for size in sizes:
        print(f"\nSize: {size:,}")

        # Create tensors
        x_nova = nova.randn(size)
        x_torch = torch.randn(size)

        # Variance
        _, t_nova = nova_var(x_nova)
        nova_var_times.append(t_nova * 1000)

        _, t_torch = torch_var(x_torch)
        torch_var_times.append(t_torch * 1000)

        print(f"  Var: NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms")

        # Standard deviation
        _, t_nova = nova_std(x_nova)
        nova_std_times.append(t_nova * 1000)

        _, t_torch = torch_std(x_torch)
        torch_std_times.append(t_torch * 1000)

        print(f"  Std: NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms")

    return {
        "sizes": sizes,
        "nova_var": nova_var_times,
        "torch_var": torch_var_times,
        "nova_std": nova_std_times,
        "torch_std": torch_std_times,
    }


def plot_basic_reductions(results):
    """Generate plots for basic reduction operations."""

    # Plot 1: Sum Performance
    fig, ax = plt.subplots()

    ax.plot(
        results["sizes"], results["nova_sum"], "o-", label="NovaNN", color=COLORS[0]
    )
    ax.plot(
        results["sizes"], results["torch_sum"], "s-", label="PyTorch", color=COLORS[1]
    )

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Sum Reduction Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sum_performance.png")
    plt.close()

    # Plot 2: Mean Performance
    fig, ax = plt.subplots()

    ax.plot(
        results["sizes"], results["nova_mean"], "o-", label="NovaNN", color=COLORS[0]
    )
    ax.plot(
        results["sizes"], results["torch_mean"], "s-", label="PyTorch", color=COLORS[1]
    )

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Mean Reduction Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_performance.png")
    plt.close()

    # Plot 3: Min/Max Performance Comparison
    fig, ax = plt.subplots()

    ax.plot(
        results["sizes"], results["nova_min"], "o-", label="NovaNN Min", color=COLORS[0]
    )
    ax.plot(
        results["sizes"],
        results["torch_min"],
        "s-",
        label="PyTorch Min",
        color=COLORS[1],
    )
    ax.plot(
        results["sizes"],
        results["nova_max"],
        "^--",
        label="NovaNN Max",
        color=COLORS[0],
        alpha=0.7,
    )
    ax.plot(
        results["sizes"],
        results["torch_max"],
        "d--",
        label="PyTorch Max",
        color=COLORS[1],
        alpha=0.7,
    )

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Min/Max Reduction Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "minmax_performance.png")
    plt.close()

    # Plot 4: All Basic Reductions Comparison (Bar Chart)
    fig, ax = plt.subplots()

    idx = 2  # Middle size (10000)
    operations = ["Sum", "Mean", "Min", "Max"]
    nova_times = [
        results["nova_sum"][idx],
        results["nova_mean"][idx],
        results["nova_min"][idx],
        results["nova_max"][idx],
    ]
    torch_times = [
        results["torch_sum"][idx],
        results["torch_mean"][idx],
        results["torch_min"][idx],
        results["torch_max"][idx],
    ]

    x_pos = np.arange(len(operations))
    width = 0.35

    ax.bar(
        x_pos - width / 2, nova_times, width, label="NovaNN", color=COLORS[0], alpha=0.8
    )
    ax.bar(
        x_pos + width / 2,
        torch_times,
        width,
        label="PyTorch",
        color=COLORS[1],
        alpha=0.8,
    )

    ax.set_xlabel("Operation")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Basic Reductions (Size={results['sizes'][idx]:,})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(operations)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "basic_reductions_comparison.png")
    plt.close()

    print(f"\n✓ Basic reduction plots saved to {OUTPUT_DIR}")


def plot_statistical_reductions(results):
    """Generate plots for statistical reduction operations."""

    # Plot 1: Variance Performance
    fig, ax = plt.subplots()

    ax.plot(
        results["sizes"], results["nova_var"], "o-", label="NovaNN", color=COLORS[0]
    )
    ax.plot(
        results["sizes"], results["torch_var"], "s-", label="PyTorch", color=COLORS[1]
    )

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Variance Reduction Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "var_performance.png")
    plt.close()

    # Plot 2: Standard Deviation Performance
    fig, ax = plt.subplots()

    ax.plot(
        results["sizes"], results["nova_std"], "o-", label="NovaNN", color=COLORS[0]
    )
    ax.plot(
        results["sizes"], results["torch_std"], "s-", label="PyTorch", color=COLORS[1]
    )

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Standard Deviation Reduction Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "std_performance.png")
    plt.close()

    # Plot 3: Statistical Reductions Bar Chart
    fig, ax = plt.subplots()

    idx = 2  # Middle size (10000)
    operations = ["Variance", "Std Dev"]
    nova_times = [
        results["nova_var"][idx],
        results["nova_std"][idx],
    ]
    torch_times = [
        results["torch_var"][idx],
        results["torch_std"][idx],
    ]

    x_pos = np.arange(len(operations))
    width = 0.35

    ax.bar(
        x_pos - width / 2, nova_times, width, label="NovaNN", color=COLORS[0], alpha=0.8
    )
    ax.bar(
        x_pos + width / 2,
        torch_times,
        width,
        label="PyTorch",
        color=COLORS[1],
        alpha=0.8,
    )

    ax.set_xlabel("Operation")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Statistical Reductions (Size={results['sizes'][idx]:,})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(operations)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "statistical_reductions_comparison.png")
    plt.close()

    print(f"✓ Statistical reduction plots saved to {OUTPUT_DIR}")


def print_summary(results_basic, results_stat):
    """Print benchmark summary statistics."""

    print("\n" + "=" * 70)
    print("REDUCTION OPERATIONS BENCHMARK RESULTS (CPU)")
    print("=" * 70)

    print("\n--- Basic Reductions (largest size) ---")
    idx = -1  # Last (largest) size
    size = results_basic["sizes"][idx]

    print(f"\nSize: {size:,} elements")

    ops = [
        ("Sum", results_basic["nova_sum"][idx], results_basic["torch_sum"][idx]),
        ("Mean", results_basic["nova_mean"][idx], results_basic["torch_mean"][idx]),
        ("Min", results_basic["nova_min"][idx], results_basic["torch_min"][idx]),
        ("Max", results_basic["nova_max"][idx], results_basic["torch_max"][idx]),
    ]

    for name, nova_time, torch_time in ops:
        ratio = torch_time / nova_time if nova_time > 0 else 0
        print(f"{name}:")
        print(f"  NovaNN:  {nova_time:.4f} ms")
        print(f"  PyTorch: {torch_time:.4f} ms")
        print(f"  Ratio:   {ratio:.2f}x")
        print()

    print("--- Statistical Reductions (largest size) ---")
    size = results_stat["sizes"][idx]

    print(f"\nSize: {size:,} elements")

    stat_ops = [
        ("Variance", results_stat["nova_var"][idx], results_stat["torch_var"][idx]),
        ("Std Dev", results_stat["nova_std"][idx], results_stat["torch_std"][idx]),
    ]

    for name, nova_time, torch_time in stat_ops:
        ratio = torch_time / nova_time if nova_time > 0 else 0
        print(f"{name}:")
        print(f"  NovaNN:  {nova_time:.4f} ms")
        print(f"  PyTorch: {torch_time:.4f} ms")
        print(f"  Ratio:   {ratio:.2f}x")
        print()

    # Calculate average ratios
    avg_sum_ratio = np.mean(
        np.array(results_basic["torch_sum"]) / np.array(results_basic["nova_sum"])
    )
    avg_mean_ratio = np.mean(
        np.array(results_basic["torch_mean"]) / np.array(results_basic["nova_mean"])
    )
    avg_var_ratio = np.mean(
        np.array(results_stat["torch_var"]) / np.array(results_stat["nova_var"])
    )
    avg_std_ratio = np.mean(
        np.array(results_stat["torch_std"]) / np.array(results_stat["nova_std"])
    )

    print("--- Average Performance Ratio (PyTorch / NovaNN) ---")
    print(f"Sum:      {avg_sum_ratio:.2f}x")
    print(f"Mean:     {avg_mean_ratio:.2f}x")
    print(f"Variance: {avg_var_ratio:.2f}x")
    print(f"Std Dev:  {avg_std_ratio:.2f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    nova.manual_seed(42)
    torch.manual_seed(42)

    print("Running reduction operations benchmarks (CPU)...")
    print("This may take a few minutes...\n")

    with Timer() as total_timer:
        print("=== Benchmarking basic reductions ===")
        with Timer() as basic_timer:
            results_basic = benchmark_basic_reductions()
        print(f"\n✓ Basic reductions benchmark completed in {basic_timer.elapsed:.2f}s")

        print("\n=== Benchmarking statistical reductions ===")
        with Timer() as stat_timer:
            results_stat = benchmark_statistical_reductions()
        print(
            f"\n✓ Statistical reductions benchmark completed in {stat_timer.elapsed:.2f}s"
        )

    print(f"\n✓ Total benchmark time: {total_timer.elapsed:.2f}s")

    # Generate plots
    with Timer() as plot_timer:
        plot_basic_reductions(results_basic)
        plot_statistical_reductions(results_stat)
    print(f"✓ Plotting completed in {plot_timer.elapsed:.2f}s")

    # Print summary
    print_summary(results_basic, results_stat)
