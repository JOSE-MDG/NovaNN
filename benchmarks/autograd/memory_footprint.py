"""
Benchmark: Memory Footprint

Measures the memory consumption of forward and backward passes across different
model sizes and batch configurations.

Comparison: NovaNN vs PyTorch memory usage
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import nova
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nova.utils.memory import MemoryTracker
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
    """Minimal MLP for memory benchmarking."""

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


def nova_forward_pass(model, x):
    """NovaNN forward pass."""
    return model(x).sum()


def nova_forward_backward(model, x):
    """NovaNN forward + backward pass."""
    y = model(x)
    loss = y.sum()
    loss.backward()
    return loss


def torch_forward_pass(model, x):
    """PyTorch forward pass."""
    return model(x).sum()


def torch_forward_backward(model, x):
    """PyTorch forward + backward pass."""
    y = model(x)
    loss = y.sum()
    loss.backward()
    return loss


def benchmark_memory_vs_depth():
    """Measure memory footprint as network depth increases."""
    depths = [2, 4, 6, 8, 10]
    batch_size = 32

    nova_fwd_mem = []
    nova_bwd_mem = []
    torch_fwd_mem = []
    torch_bwd_mem = []

    for depth in depths:
        print(f"\nDepth: {depth}")

        # NovaNN - Forward only
        model_nova = SimpleMLP("nova", depth=depth)
        x_nova = nova.randn(batch_size, 64, requires_grad=True)

        with MemoryTracker() as mem:
            with nova.no_grad():
                nova_forward_pass(model_nova, x_nova)
        nova_fwd_mem.append(mem.peak_mb)
        print(f"  NovaNN forward: {mem.peak_mb:.2f} MB")

        # NovaNN - Forward + Backward
        model_nova = SimpleMLP("nova", depth=depth)
        x_nova = nova.randn(batch_size, 64, requires_grad=True)

        with MemoryTracker() as mem:
            nova_forward_backward(model_nova, x_nova)
        nova_bwd_mem.append(mem.peak_mb)
        print(f"  NovaNN forward+backward: {mem.peak_mb:.2f} MB")

        # PyTorch - Forward only
        model_torch = SimpleMLP("torch", depth=depth)
        x_torch = torch.randn(batch_size, 64, requires_grad=True)

        with MemoryTracker() as mem:
            with torch.no_grad():
                torch_forward_pass(model_torch, x_torch)
        torch_fwd_mem.append(mem.peak_mb)
        print(f"  PyTorch forward: {mem.peak_mb:.2f} MB")

        # PyTorch - Forward + Backward
        model_torch = SimpleMLP("torch", depth=depth)
        x_torch = torch.randn(batch_size, 64, requires_grad=True)

        with MemoryTracker() as mem:
            torch_forward_backward(model_torch, x_torch)
        torch_bwd_mem.append(mem.peak_mb)
        print(f"  PyTorch forward+backward: {mem.peak_mb:.2f} MB")

    return {
        "depths": depths,
        "nova_fwd": nova_fwd_mem,
        "nova_bwd": nova_bwd_mem,
        "torch_fwd": torch_fwd_mem,
        "torch_bwd": torch_bwd_mem,
    }


def benchmark_memory_vs_batch():
    """Measure memory footprint as batch size increases."""
    batch_sizes = [8, 16, 32, 64, 128]
    depth = 4

    nova_fwd_mem = []
    nova_bwd_mem = []
    torch_fwd_mem = []
    torch_bwd_mem = []

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        # NovaNN - Forward only
        model_nova = SimpleMLP("nova", depth=depth)
        x_nova = nova.randn(batch_size, 64, requires_grad=True)

        with MemoryTracker() as mem:
            with nova.no_grad():
                nova_forward_pass(model_nova, x_nova)
        nova_fwd_mem.append(mem.peak_mb)
        print(f"  NovaNN forward: {mem.peak_mb:.2f} MB")

        # NovaNN - Forward + Backward
        model_nova = SimpleMLP("nova", depth=depth)
        x_nova = nova.randn(batch_size, 64, requires_grad=True)

        with MemoryTracker() as mem:
            nova_forward_backward(model_nova, x_nova)
        nova_bwd_mem.append(mem.peak_mb)
        print(f"  NovaNN forward+backward: {mem.peak_mb:.2f} MB")

        # PyTorch - Forward only
        model_torch = SimpleMLP("torch", depth=depth)
        x_torch = torch.randn(batch_size, 64, requires_grad=True)

        with MemoryTracker() as mem:
            with torch.no_grad():
                torch_forward_pass(model_torch, x_torch)
        torch_fwd_mem.append(mem.peak_mb)
        print(f"  PyTorch forward: {mem.peak_mb:.2f} MB")

        # PyTorch - Forward + Backward
        model_torch = SimpleMLP("torch", depth=depth)
        x_torch = torch.randn(batch_size, 64, requires_grad=True)

        with MemoryTracker() as mem:
            torch_forward_backward(model_torch, x_torch)
        torch_bwd_mem.append(mem.peak_mb)
        print(f"  PyTorch forward+backward: {mem.peak_mb:.2f} MB")

    return {
        "batch_sizes": batch_sizes,
        "nova_fwd": nova_fwd_mem,
        "nova_bwd": nova_bwd_mem,
        "torch_fwd": torch_fwd_mem,
        "torch_bwd": torch_bwd_mem,
    }


def plot_memory_comparison(results_depth, results_batch):
    """Generate publication-quality plots."""

    # Plot 1: Memory vs Depth (Framework Comparison)
    fig, ax = plt.subplots()

    ax.plot(
        results_depth["depths"],
        results_depth["nova_bwd"],
        "o-",
        label="NovaNN",
        color=COLORS[0],
    )
    ax.plot(
        results_depth["depths"],
        results_depth["torch_bwd"],
        "s-",
        label="PyTorch",
        color=COLORS[1],
    )

    ax.set_xlabel("Network Depth (layers)")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory Footprint vs Network Depth")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "memory_vs_depth.png")
    plt.close()

    # Plot 2: Forward vs Backward Memory (NovaNN)
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
        results_depth["nova_bwd"],
        width,
        label="Forward + Backward",
        color=COLORS[1],
        alpha=0.8,
    )

    ax.set_xlabel("Network Depth (layers)")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("NovaNN: Memory Usage Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_depth["depths"])
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nova_memory_fwd_vs_bwd.png")
    plt.close()

    # Plot 3: Memory Overhead (Backward - Forward)
    fig, ax = plt.subplots()

    nova_overhead = np.array(results_depth["nova_bwd"]) - np.array(
        results_depth["nova_fwd"]
    )
    torch_overhead = np.array(results_depth["torch_bwd"]) - np.array(
        results_depth["torch_fwd"]
    )

    ax.plot(
        results_depth["depths"], nova_overhead, "o-", label="NovaNN", color=COLORS[0]
    )
    ax.plot(
        results_depth["depths"], torch_overhead, "s-", label="PyTorch", color=COLORS[1]
    )

    ax.set_xlabel("Network Depth (layers)")
    ax.set_ylabel("Memory Overhead (MB)")
    ax.set_title("Backward Pass Memory Overhead")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "memory_overhead.png")
    plt.close()

    # Plot 4: Memory vs Batch Size
    fig, ax = plt.subplots()

    ax.plot(
        results_batch["batch_sizes"],
        results_batch["nova_bwd"],
        "o-",
        label="NovaNN",
        color=COLORS[0],
    )
    ax.plot(
        results_batch["batch_sizes"],
        results_batch["torch_bwd"],
        "s-",
        label="PyTorch",
        color=COLORS[1],
    )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory Footprint vs Batch Size")
    ax.legend(frameon=False)
    ax.set_xscale("log", base=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "memory_vs_batch.png")
    plt.close()

    print(f"\n✓ Plots saved to {OUTPUT_DIR}")


def print_summary(results_depth, results_batch):
    """Print benchmark summary statistics."""

    print("\n" + "=" * 60)
    print("MEMORY FOOTPRINT BENCHMARK RESULTS")
    print("=" * 60)

    print("\n--- Memory Usage vs Network Depth ---")
    for i, depth in enumerate(results_depth["depths"]):
        nova_fwd = results_depth["nova_fwd"][i]
        nova_bwd = results_depth["nova_bwd"][i]
        torch_fwd = results_depth["torch_fwd"][i]
        torch_bwd = results_depth["torch_bwd"][i]

        nova_overhead = nova_bwd - nova_fwd
        torch_overhead = torch_bwd - torch_fwd

        print(f"Depth {depth:2d}:")
        print(
            f"  NovaNN:  Fwd={nova_fwd:6.2f}MB  Bwd={nova_bwd:6.2f}MB  Overhead={nova_overhead:6.2f}MB"
        )
        print(
            f"  PyTorch: Fwd={torch_fwd:6.2f}MB  Bwd={torch_bwd:6.2f}MB  Overhead={torch_overhead:6.2f}MB"
        )

    print("\n--- Memory Usage vs Batch Size ---")
    for i, bs in enumerate(results_batch["batch_sizes"]):
        nova_mem = results_batch["nova_bwd"][i]
        torch_mem = results_batch["torch_bwd"][i]
        ratio = nova_mem / torch_mem if torch_mem > 0 else 0

        print(
            f"Batch {bs:3d}: NovaNN={nova_mem:6.2f}MB  |  PyTorch={torch_mem:6.2f}MB  |  Ratio={ratio:.2f}x"
        )

    # Average memory usage
    avg_nova = np.mean(results_depth["nova_bwd"])
    avg_torch = np.mean(results_depth["torch_bwd"])

    print("\n--- Average Memory Usage (Forward+Backward) ---")
    print(f"NovaNN:  {avg_nova:.2f} MB")
    print(f"PyTorch: {avg_torch:.2f} MB")
    print(f"Ratio:   {avg_nova/avg_torch:.2f}x")

    # Average overhead
    avg_nova_overhead = np.mean(
        np.array(results_depth["nova_bwd"]) - np.array(results_depth["nova_fwd"])
    )
    avg_torch_overhead = np.mean(
        np.array(results_depth["torch_bwd"]) - np.array(results_depth["torch_fwd"])
    )

    print("\n--- Average Backward Overhead ---")
    print(f"NovaNN:  {avg_nova_overhead:.2f} MB")
    print(f"PyTorch: {avg_torch_overhead:.2f} MB")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    nova.manual_seed(42)
    torch.manual_seed(42)

    print("Running memory footprint benchmarks...")
    print("This may take a few minutes...\n")

    with Timer() as total_timer:
        print("=== Benchmarking memory vs depth ===")
        with Timer() as depth_timer:
            results_depth = benchmark_memory_vs_depth()
        print(f"\n✓ Depth benchmark completed in {depth_timer.elapsed:.2f}s")

        print("\n=== Benchmarking memory vs batch size ===")
        with Timer() as batch_timer:
            results_batch = benchmark_memory_vs_batch()
        print(f"\n✓ Batch benchmark completed in {batch_timer.elapsed:.2f}s")

    print(f"\n✓ Total benchmark time: {total_timer.elapsed:.2f}s")

    # Generate plots
    with Timer() as plot_timer:
        plot_memory_comparison(results_depth, results_batch)
    print(f"✓ Plotting completed in {plot_timer.elapsed:.2f}s")

    # Print summary
    print_summary(results_depth, results_batch)
