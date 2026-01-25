"""
Benchmark: Gradient Accumulation

Measures the performance and correctness of gradient accumulation across multiple
micro-batches compared to single large batch processing.

Comparison: NovaNN vs PyTorch, different accumulation steps
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
OUTPUT_DIR = Path("images/benchmarks/autograd")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SimpleMLP:
    """Minimal MLP for benchmarking gradient accumulation."""

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

    def zero_grad(self):
        """Zero out gradients."""
        for layer in self.layers:
            layer.zero_grad()


@chronometer(n_iters=20, warmup=5, return_time=True, verbose=False)
def nova_single_batch(model, x, target):
    """Standard single batch forward+backward."""
    model.zero_grad()
    y = model(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    return loss


@chronometer(n_iters=20, warmup=5, return_time=True, verbose=False)
def nova_gradient_accumulation(model, x_list, target_list, accum_steps):
    """Gradient accumulation across micro-batches."""
    model.zero_grad()
    total_loss = 0

    for i in range(accum_steps):
        y = model(x_list[i])
        loss = ((y - target_list[i]) ** 2).mean()
        # Scale loss by number of accumulation steps
        scaled_loss = loss / accum_steps
        scaled_loss.backward()
        total_loss += loss.data

    return total_loss / accum_steps


@chronometer(n_iters=20, warmup=5, return_time=True, verbose=False)
def torch_single_batch(model, x, target):
    """Standard single batch forward+backward."""
    model.zero_grad()
    y = model(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    return loss


@chronometer(n_iters=20, warmup=5, return_time=True, verbose=False)
def torch_gradient_accumulation(model, x_list, target_list, accum_steps):
    """Gradient accumulation across micro-batches."""
    model.zero_grad()
    total_loss = 0

    for i in range(accum_steps):
        y = model(x_list[i])
        loss = ((y - target_list[i]) ** 2).mean()
        # Scale loss by number of accumulation steps
        scaled_loss = loss / accum_steps
        scaled_loss.backward()
        total_loss += loss.item()

    return total_loss / accum_steps


def benchmark_accumulation_steps():
    """Measure performance across different accumulation steps."""
    accumulation_steps = [1, 2, 4, 8, 16]
    micro_batch_size = 16
    depth = 4

    nova_times = []
    torch_times = []
    nova_single_times = []
    torch_single_times = []

    for accum_steps in accumulation_steps:
        print(f"\nAccumulation steps: {accum_steps}")
        effective_batch = micro_batch_size * accum_steps

        # NovaNN - Gradient Accumulation
        with Timer() as nova_timer:
            model_nova = SimpleMLP("nova", depth=depth)
            x_list = [
                nova.randn(micro_batch_size, 64, requires_grad=True)
                for _ in range(accum_steps)
            ]
            target_list = [nova.randn(micro_batch_size, 10) for _ in range(accum_steps)]

            _, accum_time = nova_gradient_accumulation(
                model_nova, x_list, target_list, accum_steps
            )
            nova_times.append(accum_time * 1000)  # to ms

        print(f"  NovaNN accumulation: {nova_timer.elapsed*1000:.2f}ms")

        # NovaNN - Single Batch (for comparison)
        with Timer() as nova_single_timer:
            model_nova_single = SimpleMLP("nova", depth=depth)
            x_single = nova.randn(effective_batch, 64, requires_grad=True)
            target_single = nova.randn(effective_batch, 10)

            _, single_time = nova_single_batch(
                model_nova_single, x_single, target_single
            )
            nova_single_times.append(single_time * 1000)

        print(f"  NovaNN single batch: {nova_single_timer.elapsed*1000:.2f}ms")

        # PyTorch - Gradient Accumulation
        with Timer() as torch_timer:
            model_torch = SimpleMLP("torch", depth=depth)
            x_list_torch = [
                torch.randn(micro_batch_size, 64, requires_grad=True)
                for _ in range(accum_steps)
            ]
            target_list_torch = [
                torch.randn(micro_batch_size, 10) for _ in range(accum_steps)
            ]

            _, accum_time = torch_gradient_accumulation(
                model_torch, x_list_torch, target_list_torch, accum_steps
            )
            torch_times.append(accum_time * 1000)

        print(f"  PyTorch accumulation: {torch_timer.elapsed*1000:.2f}ms")

        # PyTorch - Single Batch
        with Timer() as torch_single_timer:
            model_torch_single = SimpleMLP("torch", depth=depth)
            x_single_torch = torch.randn(effective_batch, 64, requires_grad=True)
            target_single_torch = torch.randn(effective_batch, 10)

            _, single_time = torch_single_batch(
                model_torch_single, x_single_torch, target_single_torch
            )
            torch_single_times.append(single_time * 1000)

        print(f"  PyTorch single batch: {torch_single_timer.elapsed*1000:.2f}ms")

    return {
        "accumulation_steps": accumulation_steps,
        "nova_accum": nova_times,
        "nova_single": nova_single_times,
        "torch_accum": torch_times,
        "torch_single": torch_single_times,
    }


def benchmark_micro_batch_size():
    """Measure performance across different micro-batch sizes."""
    micro_batch_sizes = [4, 8, 16, 32, 64]
    accum_steps = 4
    depth = 4

    nova_times = []
    torch_times = []

    for micro_batch in micro_batch_sizes:
        print(f"\nMicro-batch size: {micro_batch}")

        # NovaNN
        with Timer() as nova_timer:
            model_nova = SimpleMLP("nova", depth=depth)
            x_list = [
                nova.randn(micro_batch, 64, requires_grad=True)
                for _ in range(accum_steps)
            ]
            target_list = [nova.randn(micro_batch, 10) for _ in range(accum_steps)]

            _, accum_time = nova_gradient_accumulation(
                model_nova, x_list, target_list, accum_steps
            )
            nova_times.append(accum_time * 1000)

        print(f"  NovaNN: {nova_timer.elapsed*1000:.2f}ms")

        # PyTorch
        with Timer() as torch_timer:
            model_torch = SimpleMLP("torch", depth=depth)
            x_list_torch = [
                torch.randn(micro_batch, 64, requires_grad=True)
                for _ in range(accum_steps)
            ]
            target_list_torch = [
                torch.randn(micro_batch, 10) for _ in range(accum_steps)
            ]

            _, accum_time = torch_gradient_accumulation(
                model_torch, x_list_torch, target_list_torch, accum_steps
            )
            torch_times.append(accum_time * 1000)

        print(f"  PyTorch: {torch_timer.elapsed*1000:.2f}ms")

    return {
        "micro_batch_sizes": micro_batch_sizes,
        "nova_times": nova_times,
        "torch_times": torch_times,
    }


@chronometer
def plot_accumulation_comparison(results_steps, results_micro):
    """Generate publication-quality plots."""

    # Plot 1: Accumulation vs Single Batch (NovaNN)
    fig, ax = plt.subplots()

    x_pos = np.arange(len(results_steps["accumulation_steps"]))
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        results_steps["nova_single"],
        width,
        label="Single Batch",
        color=COLORS[0],
        alpha=0.8,
    )
    ax.bar(
        x_pos + width / 2,
        results_steps["nova_accum"],
        width,
        label="Gradient Accumulation",
        color=COLORS[1],
        alpha=0.8,
    )

    ax.set_xlabel("Accumulation Steps")
    ax.set_ylabel("Time (ms)")
    ax.set_title("NovaNN: Single Batch vs Gradient Accumulation")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_steps["accumulation_steps"])
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nova_accumulation_comparison.png")
    plt.close()

    # Plot 2: Framework Comparison - Accumulation
    fig, ax = plt.subplots()

    ax.plot(
        results_steps["accumulation_steps"],
        results_steps["nova_accum"],
        "o-",
        label="NovaNN",
        color=COLORS[0],
    )
    ax.plot(
        results_steps["accumulation_steps"],
        results_steps["torch_accum"],
        "s-",
        label="PyTorch",
        color=COLORS[1],
    )

    ax.set_xlabel("Accumulation Steps")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Gradient Accumulation Performance")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accumulation_framework_comparison.png")
    plt.close()

    # Plot 3: Overhead of Accumulation
    fig, ax = plt.subplots()

    nova_overhead = (
        np.array(results_steps["nova_accum"]) / np.array(results_steps["nova_single"])
        - 1
    ) * 100
    torch_overhead = (
        np.array(results_steps["torch_accum"]) / np.array(results_steps["torch_single"])
        - 1
    ) * 100

    ax.plot(
        results_steps["accumulation_steps"],
        nova_overhead,
        "o-",
        label="NovaNN",
        color=COLORS[0],
    )
    ax.plot(
        results_steps["accumulation_steps"],
        torch_overhead,
        "s-",
        label="PyTorch",
        color=COLORS[1],
    )

    ax.set_xlabel("Accumulation Steps")
    ax.set_ylabel("Overhead (%)")
    ax.set_title("Gradient Accumulation Overhead vs Single Batch")
    ax.legend(frameon=False)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accumulation_overhead.png")
    plt.close()

    # Plot 4: Scaling with Micro-Batch Size
    fig, ax = plt.subplots()

    ax.plot(
        results_micro["micro_batch_sizes"],
        results_micro["nova_times"],
        "o-",
        label="NovaNN",
        color=COLORS[0],
    )
    ax.plot(
        results_micro["micro_batch_sizes"],
        results_micro["torch_times"],
        "s-",
        label="PyTorch",
        color=COLORS[1],
    )

    ax.set_xlabel("Micro-Batch Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Gradient Accumulation vs Micro-Batch Size")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accumulation_vs_microbatch.png")
    plt.close()

    print(f"\n✓ Plots saved to {OUTPUT_DIR}")


def print_summary(results_steps, results_micro):
    """Print benchmark summary statistics."""

    print("\n" + "=" * 60)
    print("GRADIENT ACCUMULATION BENCHMARK RESULTS")
    print("=" * 60)

    print("\n--- Performance vs Accumulation Steps ---")
    for i, steps in enumerate(results_steps["accumulation_steps"]):
        eff_batch = steps * 16
        nova_accum = results_steps["nova_accum"][i]
        nova_single = results_steps["nova_single"][i]
        torch_accum = results_steps["torch_accum"][i]
        torch_single = results_steps["torch_single"][i]

        nova_overhead = ((nova_accum / nova_single) - 1) * 100
        torch_overhead = ((torch_accum / torch_single) - 1) * 100

        print(f"Steps {steps:2d} (batch={eff_batch:3d}):")
        print(
            f"  NovaNN:  Accum={nova_accum:6.2f}ms  Single={nova_single:6.2f}ms  Overhead={nova_overhead:+6.1f}%"
        )
        print(
            f"  PyTorch: Accum={torch_accum:6.2f}ms  Single={torch_single:6.2f}ms  Overhead={torch_overhead:+6.1f}%"
        )

    print("\n--- Performance vs Micro-Batch Size ---")
    for i, micro_batch in enumerate(results_micro["micro_batch_sizes"]):
        eff_batch = micro_batch * 4
        nova_time = results_micro["nova_times"][i]
        torch_time = results_micro["torch_times"][i]
        ratio = nova_time / torch_time

        print(
            f"Micro-batch {micro_batch:2d} (total={eff_batch:3d}): "
            f"NovaNN={nova_time:6.2f}ms  |  PyTorch={torch_time:6.2f}ms  |  Ratio={ratio:.2f}x"
        )

    # Average overhead
    avg_nova_overhead = np.mean(
        (
            np.array(results_steps["nova_accum"])
            / np.array(results_steps["nova_single"])
            - 1
        )
        * 100
    )
    avg_torch_overhead = np.mean(
        (
            np.array(results_steps["torch_accum"])
            / np.array(results_steps["torch_single"])
            - 1
        )
        * 100
    )

    print("\n--- Average Accumulation Overhead ---")
    print(f"NovaNN:  {avg_nova_overhead:+.1f}%")
    print(f"PyTorch: {avg_torch_overhead:+.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    nova.manual_seed(42)
    torch.manual_seed(42)

    print("Running gradient accumulation benchmarks...")
    print("This may take a few minutes...\n")

    # Run benchmarks with Timer for total execution time
    with Timer() as total_timer:
        print("=== Benchmarking accumulation steps ===")
        with Timer() as steps_timer:
            results_steps = benchmark_accumulation_steps()
        print(
            f"\n✓ Accumulation steps benchmark completed in {steps_timer.elapsed:.2f}s"
        )

        print("\n=== Benchmarking micro-batch sizes ===")
        with Timer() as micro_timer:
            results_micro = benchmark_micro_batch_size()
        print(f"\n✓ Micro-batch benchmark completed in {micro_timer.elapsed:.2f}s")

    print(f"\n✓ Total benchmark time: {total_timer.elapsed:.2f}s")

    # Generate plots (timed with chronometer)
    plot_accumulation_comparison(results_steps, results_micro)

    # Print summary
    print_summary(results_steps, results_micro)
