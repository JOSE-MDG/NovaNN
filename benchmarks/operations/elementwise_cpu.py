"""
Benchmark: Element-wise Operations (CPU)

Measures the performance of element-wise operations (addition, multiplication,
activation functions) on CPU.

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
def nova_add(a, b):
    """NovaNN element-wise addition."""
    return a + b


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_add(a, b):
    """PyTorch element-wise addition."""
    return a + b


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def nova_mul(a, b):
    """NovaNN element-wise multiplication."""
    return a * b


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_mul(a, b):
    """PyTorch element-wise multiplication."""
    return a * b


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def nova_relu(x):
    """NovaNN ReLU activation."""
    return nova.nn.functional.relu(x)


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_relu(x):
    """PyTorch ReLU activation."""
    return torch.nn.functional.relu(x)


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def nova_sigmoid(x):
    """NovaNN Sigmoid activation."""
    return nova.nn.functional.sigmoid(x)


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_sigmoid(x):
    """PyTorch Sigmoid activation."""
    return torch.sigmoid(x)


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def nova_tanh(x):
    """NovaNN Tanh activation."""
    return nova.nn.functional.tanh(x)


@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def torch_tanh(x):
    """PyTorch Tanh activation."""
    return torch.tanh(x)


def benchmark_arithmetic_ops():
    """Benchmark basic arithmetic operations across different sizes."""
    sizes = [100, 1000, 10000, 100000, 1000000]

    nova_add_times = []
    torch_add_times = []

    nova_mul_times = []
    torch_mul_times = []

    for size in sizes:
        print(f"\nSize: {size:,}")

        # Create tensors
        a_nova = nova.randn(size)
        b_nova = nova.randn(size)
        a_torch = torch.randn(size)
        b_torch = torch.randn(size)

        # Addition
        _, t_nova = nova_add(a_nova, b_nova)
        nova_add_times.append(t_nova * 1000)

        _, t_torch = torch_add(a_torch, b_torch)
        torch_add_times.append(t_torch * 1000)

        print(f"  Addition: NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms")

        # Multiplication
        _, t_nova = nova_mul(a_nova, b_nova)
        nova_mul_times.append(t_nova * 1000)

        _, t_torch = torch_mul(a_torch, b_torch)
        torch_mul_times.append(t_torch * 1000)

        print(
            f"  Multiplication: NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms"
        )

    return {
        "sizes": sizes,
        "nova_add": nova_add_times,
        "torch_add": torch_add_times,
        "nova_mul": nova_mul_times,
        "torch_mul": torch_mul_times,
    }


def benchmark_activation_ops():
    """Benchmark activation functions across different sizes."""
    sizes = [100, 1000, 10000, 100000, 1000000]

    nova_relu_times = []
    torch_relu_times = []

    nova_sigmoid_times = []
    torch_sigmoid_times = []

    nova_tanh_times = []
    torch_tanh_times = []

    for size in sizes:
        print(f"\nSize: {size:,}")

        # Create tensors
        x_nova = nova.randn(size)
        x_torch = torch.randn(size)

        # ReLU
        _, t_nova = nova_relu(x_nova)
        nova_relu_times.append(t_nova * 1000)

        _, t_torch = torch_relu(x_torch)
        torch_relu_times.append(t_torch * 1000)

        print(f"  ReLU: NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms")

        # Sigmoid
        _, t_nova = nova_sigmoid(x_nova)
        nova_sigmoid_times.append(t_nova * 1000)

        _, t_torch = torch_sigmoid(x_torch)
        torch_sigmoid_times.append(t_torch * 1000)

        print(f"  Sigmoid: NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms")

        # Tanh
        _, t_nova = nova_tanh(x_nova)
        nova_tanh_times.append(t_nova * 1000)

        _, t_torch = torch_tanh(x_torch)
        torch_tanh_times.append(t_torch * 1000)

        print(f"  Tanh: NovaNN={t_nova*1000:.4f}ms  PyTorch={t_torch*1000:.4f}ms")

    return {
        "sizes": sizes,
        "nova_relu": nova_relu_times,
        "torch_relu": torch_relu_times,
        "nova_sigmoid": nova_sigmoid_times,
        "torch_sigmoid": torch_sigmoid_times,
        "nova_tanh": nova_tanh_times,
        "torch_tanh": torch_tanh_times,
    }


def plot_arithmetic_comparison(results):
    """Generate plots for arithmetic operations."""

    # Plot 1: Addition Performance
    fig, ax = plt.subplots()

    ax.plot(
        results["sizes"], results["nova_add"], "o-", label="NovaNN", color=COLORS[0]
    )
    ax.plot(
        results["sizes"], results["torch_add"], "s-", label="PyTorch", color=COLORS[1]
    )

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Element-wise Addition Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "addition_performance.png")
    plt.close()

    # Plot 2: Multiplication Performance
    fig, ax = plt.subplots()

    ax.plot(
        results["sizes"], results["nova_mul"], "o-", label="NovaNN", color=COLORS[0]
    )
    ax.plot(
        results["sizes"], results["torch_mul"], "s-", label="PyTorch", color=COLORS[1]
    )

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Element-wise Multiplication Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multiplication_performance.png")
    plt.close()

    # Plot 3: Speedup Ratio
    fig, ax = plt.subplots()

    add_ratio = np.array(results["torch_add"]) / np.array(results["nova_add"])
    mul_ratio = np.array(results["torch_mul"]) / np.array(results["nova_mul"])

    ax.plot(results["sizes"], add_ratio, "o-", label="Addition", color=COLORS[0])
    ax.plot(results["sizes"], mul_ratio, "s-", label="Multiplication", color=COLORS[1])

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Speedup (PyTorch / NovaNN)")
    ax.set_title("Relative Performance")
    ax.set_xscale("log")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arithmetic_speedup.png")
    plt.close()

    print(f"\n✓ Arithmetic plots saved to {OUTPUT_DIR}")


def plot_activation_comparison(results):
    """Generate plots for activation functions."""

    # Plot 1: ReLU Performance
    fig, ax = plt.subplots()

    ax.plot(
        results["sizes"], results["nova_relu"], "o-", label="NovaNN", color=COLORS[0]
    )
    ax.plot(
        results["sizes"], results["torch_relu"], "s-", label="PyTorch", color=COLORS[1]
    )

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("ReLU Activation Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "relu_performance.png")
    plt.close()

    # Plot 2: Sigmoid Performance
    fig, ax = plt.subplots()

    ax.plot(
        results["sizes"], results["nova_sigmoid"], "o-", label="NovaNN", color=COLORS[0]
    )
    ax.plot(
        results["sizes"],
        results["torch_sigmoid"],
        "s-",
        label="PyTorch",
        color=COLORS[1],
    )

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Sigmoid Activation Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sigmoid_performance.png")
    plt.close()

    # Plot 3: Activation Comparison Bar Chart
    fig, ax = plt.subplots()

    # Use middle size (10000) for comparison
    idx = 2
    operations = ["ReLU", "Sigmoid", "Tanh"]
    nova_times = [
        results["nova_relu"][idx],
        results["nova_sigmoid"][idx],
        results["nova_tanh"][idx],
    ]
    torch_times = [
        results["torch_relu"][idx],
        results["torch_sigmoid"][idx],
        results["torch_tanh"][idx],
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

    ax.set_xlabel("Activation Function")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Activation Functions (Size={results['sizes'][idx]:,})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(operations)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "activation_comparison.png")
    plt.close()

    print(f"✓ Activation plots saved to {OUTPUT_DIR}")


def print_summary(results_arith, results_act):
    """Print benchmark summary statistics."""

    print("\n" + "=" * 70)
    print("ELEMENT-WISE OPERATIONS BENCHMARK RESULTS (CPU)")
    print("=" * 70)

    print("\n--- Arithmetic Operations (largest size) ---")
    idx = -1  # Last (largest) size
    size = results_arith["sizes"][idx]

    print(f"\nSize: {size:,} elements")
    print("Addition:")
    print(f"  NovaNN:  {results_arith['nova_add'][idx]:.4f} ms")
    print(f"  PyTorch: {results_arith['torch_add'][idx]:.4f} ms")
    print(
        f"  Ratio:   {results_arith['torch_add'][idx] / results_arith['nova_add'][idx]:.2f}x"
    )

    print("\nMultiplication:")
    print(f"  NovaNN:  {results_arith['nova_mul'][idx]:.4f} ms")
    print(f"  PyTorch: {results_arith['torch_mul'][idx]:.4f} ms")
    print(
        f"  Ratio:   {results_arith['torch_mul'][idx] / results_arith['nova_mul'][idx]:.2f}x"
    )

    print("\n--- Activation Functions (largest size) ---")
    size = results_act["sizes"][idx]

    print(f"\nSize: {size:,} elements")
    print("ReLU:")
    print(f"  NovaNN:  {results_act['nova_relu'][idx]:.4f} ms")
    print(f"  PyTorch: {results_act['torch_relu'][idx]:.4f} ms")
    print(
        f"  Ratio:   {results_act['torch_relu'][idx] / results_act['nova_relu'][idx]:.2f}x"
    )

    print("\nSigmoid:")
    print(f"  NovaNN:  {results_act['nova_sigmoid'][idx]:.4f} ms")
    print(f"  PyTorch: {results_act['torch_sigmoid'][idx]:.4f} ms")
    print(
        f"  Ratio:   {results_act['torch_sigmoid'][idx] / results_act['nova_sigmoid'][idx]:.2f}x"
    )

    print("\nTanh:")
    print(f"  NovaNN:  {results_act['nova_tanh'][idx]:.4f} ms")
    print(f"  PyTorch: {results_act['torch_tanh'][idx]:.4f} ms")
    print(
        f"  Ratio:   {results_act['torch_tanh'][idx] / results_act['nova_tanh'][idx]:.2f}x"
    )

    # Calculate average ratios
    avg_add_ratio = np.mean(
        np.array(results_arith["torch_add"]) / np.array(results_arith["nova_add"])
    )
    avg_mul_ratio = np.mean(
        np.array(results_arith["torch_mul"]) / np.array(results_arith["nova_mul"])
    )
    avg_relu_ratio = np.mean(
        np.array(results_act["torch_relu"]) / np.array(results_act["nova_relu"])
    )

    print("\n--- Average Performance Ratio (PyTorch / NovaNN) ---")
    print(f"Addition:        {avg_add_ratio:.2f}x")
    print(f"Multiplication:  {avg_mul_ratio:.2f}x")
    print(f"ReLU:            {avg_relu_ratio:.2f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    nova.manual_seed(42)
    torch.manual_seed(42)

    print("Running element-wise operations benchmarks (CPU)...")
    print("This may take a few minutes...\n")

    with Timer() as total_timer:
        print("=== Benchmarking arithmetic operations ===")
        with Timer() as arith_timer:
            results_arith = benchmark_arithmetic_ops()
        print(f"\n✓ Arithmetic benchmark completed in {arith_timer.elapsed:.2f}s")

        print("\n=== Benchmarking activation functions ===")
        with Timer() as act_timer:
            results_act = benchmark_activation_ops()
        print(f"\n✓ Activation benchmark completed in {act_timer.elapsed:.2f}s")

    print(f"\n✓ Total benchmark time: {total_timer.elapsed:.2f}s")

    # Generate plots
    with Timer() as plot_timer:
        plot_arithmetic_comparison(results_arith)
        plot_activation_comparison(results_act)
    print(f"✓ Plotting completed in {plot_timer.elapsed:.2f}s")

    # Print summary
    print_summary(results_arith, results_act)
