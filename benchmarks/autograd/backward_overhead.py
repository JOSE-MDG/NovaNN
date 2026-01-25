"""
Benchmark: Autograd Backward Overhead

Measures the overhead introduced by the autograd system when computing gradients
compared to forward-only operations.

Comparison: NovaNN vs PyTorch, with/without gradients
"""

# Resolve imports ..uitls

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
OUTPUT_DIR = Path("images/benchmarks/autograd")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SimpleMLP:
    """Minimal MLP for benchmarking."""

    def __init__(self, framework="nova", depth=3, width=128):
        self.framework = framework
        self.depth = depth

        if framework == "nova":
            self.layers = []
            in_size = 64
            for _ in range(depth):
                self.layers.append(nova.nn.Linear(in_size, width))
                in_size = width
            self.layers.append(nova.nn.Linear(width, 10))
        else:  # torch
            self.layers = []
            in_size = 64
            for _ in range(depth):
                self.layers.append(torch.nn.Linear(in_size, width))
                in_size = width
            self.layers.append(torch.nn.Linear(width, 10))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.framework == "nova":
                x = nova.nn.functional.relu(x)
            else:
                x = torch.nn.functional.relu(x)
        return self.layers[-1](x)


@chronometer(n_iters=50, warmup=5, return_time=True, verbose=False)
def nova_forward_only(model, x):
    """Forward pass without gradient tracking."""
    with nova.no_grad():
        y = model(x)
        return y.sum()


@chronometer(n_iters=50, warmup=5, return_time=True, verbose=False)
def nova_forward_backward(model, x):
    """Forward + backward pass."""
    y = model(x)
    loss = y.sum()
    loss.backward()
    return loss


@chronometer(n_iters=50, warmup=5, return_time=True, verbose=False)
def torch_forward_only(model, x):
    """Forward pass without gradient tracking."""
    with torch.no_grad():
        y = model(x)
        return y.sum()


@chronometer(n_iters=50, warmup=5, return_time=True, verbose=False)
def torch_forward_backward(model, x):
    """Forward + backward pass."""
    y = model(x)
    loss = y.sum()
    loss.backward()
    return loss


def benchmark_overhead_vs_depth():
    """Measure overhead as network depth increases."""
    depths = [2, 4, 6, 8, 10]
    batch_size = 32

    nova_fwd_times = []
    nova_fwd_bwd_times = []
    torch_fwd_times = []
    torch_fwd_bwd_times = []

    for depth in depths:
        print(f"\nDepth: {depth}")

        # NovaNN
        with Timer() as nova_timer:
            model_nova = SimpleMLP("nova", depth=depth)
            x_nova = nova.randn(batch_size, 64, requires_grad=True)

            _, fwd_time = nova_forward_only(model_nova, x_nova)
            nova_fwd_times.append(fwd_time * 1000)  # to ms

            _, fwd_bwd_time = nova_forward_backward(model_nova, x_nova)
            nova_fwd_bwd_times.append(fwd_bwd_time * 1000)

        print(f"  NovaNN total: {nova_timer.elapsed*1000:.2f}ms")

        # PyTorch
        with Timer() as torch_timer:
            model_torch = SimpleMLP("torch", depth=depth)
            x_torch = torch.randn(batch_size, 64, requires_grad=True)

            _, fwd_time = torch_forward_only(model_torch, x_torch)
            torch_fwd_times.append(fwd_time * 1000)

            _, fwd_bwd_time = torch_forward_backward(model_torch, x_torch)
            torch_fwd_bwd_times.append(fwd_bwd_time * 1000)

        print(f"  PyTorch total: {torch_timer.elapsed*1000:.2f}ms")

    return {
        "depths": depths,
        "nova_fwd": nova_fwd_times,
        "nova_fwd_bwd": nova_fwd_bwd_times,
        "torch_fwd": torch_fwd_times,
        "torch_fwd_bwd": torch_fwd_bwd_times,
    }


def benchmark_overhead_vs_batch():
    """Measure overhead as batch size increases."""
    batch_sizes = [8, 16, 32, 64, 128]
    depth = 4

    nova_fwd_times = []
    nova_fwd_bwd_times = []
    torch_fwd_times = []
    torch_fwd_bwd_times = []

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        # NovaNN
        with Timer() as nova_timer:
            model_nova = SimpleMLP("nova", depth=depth)
            x_nova = nova.randn(batch_size, 64, requires_grad=True)

            _, fwd_time = nova_forward_only(model_nova, x_nova)
            nova_fwd_times.append(fwd_time * 1000)

            _, fwd_bwd_time = nova_forward_backward(model_nova, x_nova)
            nova_fwd_bwd_times.append(fwd_bwd_time * 1000)

        print(f"  NovaNN total: {nova_timer.elapsed*1000:.2f}ms")

        # PyTorch
        with Timer() as torch_timer:
            model_torch = SimpleMLP("torch", depth=depth)
            x_torch = torch.randn(batch_size, 64, requires_grad=True)

            _, fwd_time = torch_forward_only(model_torch, x_torch)
            torch_fwd_times.append(fwd_time * 1000)

            _, fwd_bwd_time = torch_forward_backward(model_torch, x_torch)
            torch_fwd_bwd_times.append(fwd_bwd_time * 1000)

        print(f"  PyTorch total: {torch_timer.elapsed*1000:.2f}ms")

    return {
        "batch_sizes": batch_sizes,
        "nova_fwd": nova_fwd_times,
        "nova_fwd_bwd": nova_fwd_bwd_times,
        "torch_fwd": torch_fwd_times,
        "torch_fwd_bwd": torch_fwd_bwd_times,
    }


@chronometer
def plot_overhead_comparison(results_depth, results_batch):
    """Generate publication-quality plots."""

    # Plot 1: Overhead vs Depth
    fig, ax = plt.subplots()

    nova_overhead = np.array(results_depth["nova_fwd_bwd"]) - np.array(
        results_depth["nova_fwd"]
    )
    torch_overhead = np.array(results_depth["torch_fwd_bwd"]) - np.array(
        results_depth["torch_fwd"]
    )

    ax.plot(
        results_depth["depths"], nova_overhead, "o-", label="NovaNN", color=COLORS[0]
    )
    ax.plot(
        results_depth["depths"], torch_overhead, "s-", label="PyTorch", color=COLORS[1]
    )
    ax.set_xlabel("Network Depth (layers)")
    ax.set_ylabel("Backward Overhead (ms)")
    ax.set_title("Autograd Overhead vs Network Depth")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overhead_vs_depth.png")
    plt.close()

    # Plot 2: Forward vs Forward+Backward (NovaNN)
    fig, ax = plt.subplots()

    x_pos = np.arange(len(results_depth["depths"]))
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        results_depth["nova_fwd"],
        width,
        label="Forward Only",
        color=COLORS[0],
        alpha=0.8,
    )
    ax.bar(
        x_pos + width / 2,
        results_depth["nova_fwd_bwd"],
        width,
        label="Forward + Backward",
        color=COLORS[1],
        alpha=0.8,
    )

    ax.set_xlabel("Network Depth (layers)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("NovaNN: Forward vs Forward+Backward")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_depth["depths"])
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nova_fwd_vs_bwd.png")
    plt.close()

    # Plot 3: Relative Overhead Percentage
    fig, ax = plt.subplots()

    nova_overhead_pct = (nova_overhead / np.array(results_depth["nova_fwd"])) * 100
    torch_overhead_pct = (torch_overhead / np.array(results_depth["torch_fwd"])) * 100

    ax.plot(
        results_depth["depths"],
        nova_overhead_pct,
        "o-",
        label="NovaNN",
        color=COLORS[0],
    )
    ax.plot(
        results_depth["depths"],
        torch_overhead_pct,
        "s-",
        label="PyTorch",
        color=COLORS[1],
    )
    ax.set_xlabel("Network Depth (layers)")
    ax.set_ylabel("Relative Overhead (%)")
    ax.set_title("Backward Pass Overhead (% of Forward Time)")
    ax.legend(frameon=False)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "relative_overhead.png")
    plt.close()

    # Plot 4: Scaling with Batch Size
    fig, ax = plt.subplots()

    nova_overhead_batch = np.array(results_batch["nova_fwd_bwd"]) - np.array(
        results_batch["nova_fwd"]
    )
    torch_overhead_batch = np.array(results_batch["torch_fwd_bwd"]) - np.array(
        results_batch["torch_fwd"]
    )

    ax.plot(
        results_batch["batch_sizes"],
        nova_overhead_batch,
        "o-",
        label="NovaNN",
        color=COLORS[0],
    )
    ax.plot(
        results_batch["batch_sizes"],
        torch_overhead_batch,
        "s-",
        label="PyTorch",
        color=COLORS[1],
    )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Backward Overhead (ms)")
    ax.set_title("Autograd Overhead vs Batch Size")
    ax.legend(frameon=False)
    ax.set_xscale("log", base=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overhead_vs_batch.png")
    plt.close()

    print(f"\n✓ Plots saved to {OUTPUT_DIR}")


def print_summary(results_depth, results_batch):
    """Print benchmark summary statistics."""

    print("\n" + "=" * 60)
    print("BACKWARD OVERHEAD BENCHMARK RESULTS")
    print("=" * 60)

    print("\n--- Overhead vs Network Depth ---")
    for i, depth in enumerate(results_depth["depths"]):
        nova_oh = results_depth["nova_fwd_bwd"][i] - results_depth["nova_fwd"][i]
        torch_oh = results_depth["torch_fwd_bwd"][i] - results_depth["torch_fwd"][i]
        nova_pct = (nova_oh / results_depth["nova_fwd"][i]) * 100
        torch_pct = (torch_oh / results_depth["torch_fwd"][i]) * 100

        print(
            f"Depth {depth:2d}: NovaNN={nova_oh:6.2f}ms ({nova_pct:5.1f}%)  |  "
            f"PyTorch={torch_oh:6.2f}ms ({torch_pct:5.1f}%)"
        )

    print("\n--- Overhead vs Batch Size ---")
    for i, bs in enumerate(results_batch["batch_sizes"]):
        nova_oh = results_batch["nova_fwd_bwd"][i] - results_batch["nova_fwd"][i]
        torch_oh = results_batch["torch_fwd_bwd"][i] - results_batch["torch_fwd"][i]

        print(f"Batch {bs:3d}: NovaNN={nova_oh:6.2f}ms  |  PyTorch={torch_oh:6.2f}ms")

    # Average overhead
    avg_nova = np.mean(
        np.array(results_depth["nova_fwd_bwd"]) - np.array(results_depth["nova_fwd"])
    )
    avg_torch = np.mean(
        np.array(results_depth["torch_fwd_bwd"]) - np.array(results_depth["torch_fwd"])
    )

    print("\n--- Average Overhead ---")
    print(f"NovaNN:  {avg_nova:.2f}ms")
    print(f"PyTorch: {avg_torch:.2f}ms")
    print(f"Ratio:   {avg_nova/avg_torch:.2f}x")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    nova.manual_seed(42)
    torch.manual_seed(42)

    print("Running backward overhead benchmarks...")
    print("This may take a few minutes...\n")

    # Run benchmarks with Timer for total execution time
    with Timer() as total_timer:
        print("=== Benchmarking overhead vs depth ===")
        with Timer() as depth_timer:
            results_depth = benchmark_overhead_vs_depth()
        print(f"\n✓ Depth benchmark completed in {depth_timer.elapsed:.2f}s")

        print("\n=== Benchmarking overhead vs batch size ===")
        with Timer() as batch_timer:
            results_batch = benchmark_overhead_vs_batch()
        print(f"\n✓ Batch benchmark completed in {batch_timer.elapsed:.2f}s")

    print(f"\n✓ Total benchmark time: {total_timer.elapsed:.2f}s")

    # Generate plots (timed with chronometer)
    plot_overhead_comparison(results_depth, results_batch)

    # Print summary
    print_summary(results_depth, results_batch)
