"""
Benchmark: End-to-End Training (CPU)

Measures the complete training pipeline performance including forward pass,
loss computation, backward pass, and optimizer step on CPU.

Comparison: NovaNN vs PyTorch
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import nova
import torch
import numpy as np
import nova.nn as nn
import nova.optim as optim
import torch.nn as torch_nn
import nova.nn.functional as F
import torch.optim as torch_optim
import matplotlib.pyplot as plt
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
OUTPUT_DIR = Path("images/benchmarks/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class NovaMLP(nn.Module):
    """Simple MLP in NovaNN."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TorchMLP(torch_nn.Module):
    """Simple MLP in PyTorch."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch_nn.Linear(input_size, hidden_size)
        self.fc2 = torch_nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch_nn.Linear(hidden_size, output_size)
        self.relu = torch_nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NovaConvNet(nn.Module):
    """Simple ConvNet in NovaNN."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TorchConvNet(torch_nn.Module):
    """Simple ConvNet in PyTorch."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch_nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = torch_nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = torch_nn.MaxPool2d(2, 2)
        self.fc1 = torch_nn.Linear(32 * 7 * 7, 128)
        self.fc2 = torch_nn.Linear(128, 10)
        self.relu = torch_nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@chronometer(n_iters=30, warmup=5, return_time=True, verbose=False)
def nova_train_step_mlp(model, optimizer, x, y):
    """Single training step in NovaNN for MLP."""
    pred = model(x)
    loss = F.cross_entropy(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


@chronometer(n_iters=30, warmup=5, return_time=True, verbose=False)
def torch_train_step_mlp(model, optimizer, x, y):
    """Single training step in PyTorch for MLP."""
    pred = model(x)
    loss = torch_nn.functional.cross_entropy(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


@chronometer(n_iters=20, warmup=3, return_time=True, verbose=False)
def nova_train_step_conv(model, optimizer, x, y):
    """Single training step in NovaNN for ConvNet."""
    pred = model(x)
    loss = F.cross_entropy(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


@chronometer(n_iters=20, warmup=3, return_time=True, verbose=False)
def torch_train_step_conv(model, optimizer, x, y):
    """Single training step in PyTorch for ConvNet."""
    pred = model(x)
    loss = torch_nn.functional.cross_entropy(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def benchmark_mlp_training(batch_sizes, hidden_size=128):
    """Benchmark MLP training across different batch sizes."""
    input_size = 784
    output_size = 10

    results = {
        "batch_sizes": batch_sizes,
        "nova_times": [],
        "torch_times": [],
    }

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        x_nova = nova.randn(batch_size, input_size)
        y_nova = nova.randint(0, output_size, (batch_size,))
        x_torch = torch.randn(batch_size, input_size)
        y_torch = torch.randint(0, output_size, (batch_size,))

        nova_model = NovaMLP(input_size, hidden_size, output_size)
        nova_optimizer = optim.SGD(nova_model.parameters(), lr=0.01)

        _, t_nova = nova_train_step_mlp(nova_model, nova_optimizer, x_nova, y_nova)
        results["nova_times"].append(t_nova * 1000)

        torch_model = TorchMLP(input_size, hidden_size, output_size)
        torch_optimizer = torch_optim.SGD(torch_model.parameters(), lr=0.01)

        _, t_torch = torch_train_step_mlp(
            torch_model, torch_optimizer, x_torch, y_torch
        )
        results["torch_times"].append(t_torch * 1000)

        print(f"  NovaNN:  {t_nova*1000:.4f} ms/step")
        print(f"  PyTorch: {t_torch*1000:.4f} ms/step")
        print(f"  Ratio:   {t_torch / t_nova:.2f}x")

    return results


def benchmark_convnet_training(batch_sizes):
    """Benchmark ConvNet training across different batch sizes."""
    results = {
        "batch_sizes": batch_sizes,
        "nova_times": [],
        "torch_times": [],
    }

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        x_nova = nova.randn(batch_size, 1, 28, 28)
        y_nova = nova.randint(0, 10, (batch_size,))
        x_torch = torch.randn(batch_size, 1, 28, 28)
        y_torch = torch.randint(0, 10, (batch_size,))

        nova_model = NovaConvNet()
        nova_optimizer = optim.Adam(nova_model.parameters(), lr=0.001)

        _, t_nova = nova_train_step_conv(nova_model, nova_optimizer, x_nova, y_nova)
        results["nova_times"].append(t_nova * 1000)

        torch_model = TorchConvNet()
        torch_optimizer = torch_optim.Adam(torch_model.parameters(), lr=0.001)

        _, t_torch = torch_train_step_conv(
            torch_model, torch_optimizer, x_torch, y_torch
        )
        results["torch_times"].append(t_torch * 1000)

        print(f"  NovaNN:  {t_nova*1000:.4f} ms/step")
        print(f"  PyTorch: {t_torch*1000:.4f} ms/step")
        print(f"  Ratio:   {t_torch / t_nova:.2f}x")

    return results


def benchmark_optimizer_comparison(batch_size=64, hidden_size=128):
    """Compare different optimizers in end-to-end training."""
    input_size = 784
    output_size = 10

    optimizers_config = [
        ("SGD", {"lr": 0.01}),
        ("Adam", {"lr": 0.001}),
        ("AdamW", {"lr": 0.001}),
        ("RMSprop", {"lr": 0.001}),
    ]

    results = {
        "optimizers": [name for name, _ in optimizers_config],
        "nova_times": [],
        "torch_times": [],
    }

    x_nova = nova.randn(batch_size, input_size)
    y_nova = nova.randint(0, output_size, (batch_size,))
    x_torch = torch.randn(batch_size, input_size)
    y_torch = torch.randint(0, output_size, (batch_size,))

    for opt_name, opt_kwargs in optimizers_config:
        print(f"\nOptimizer: {opt_name}")

        nova_model = NovaMLP(input_size, hidden_size, output_size)
        nova_opt_class = getattr(optim, opt_name)
        nova_opt = nova_opt_class(nova_model.parameters(), **opt_kwargs)

        _, t_nova = nova_train_step_mlp(nova_model, nova_opt, x_nova, y_nova)
        results["nova_times"].append(t_nova * 1000)

        torch_model = TorchMLP(input_size, hidden_size, output_size)
        torch_opt_class = getattr(torch_optim, opt_name)
        torch_opt = torch_opt_class(torch_model.parameters(), **opt_kwargs)

        _, t_torch = torch_train_step_mlp(torch_model, torch_opt, x_torch, y_torch)
        results["torch_times"].append(t_torch * 1000)

        print(f"  NovaNN:  {t_nova*1000:.4f} ms/step")
        print(f"  PyTorch: {t_torch*1000:.4f} ms/step")
        print(f"  Ratio:   {t_torch / t_nova:.2f}x")

    return results


def plot_mlp_results(results):
    """Plot MLP training benchmark results."""

    fig, ax = plt.subplots()
    ax.plot(
        results["batch_sizes"],
        results["nova_times"],
        "o-",
        label="NovaNN",
        color=COLORS[0],
    )
    ax.plot(
        results["batch_sizes"],
        results["torch_times"],
        "s-",
        label="PyTorch",
        color=COLORS[1],
    )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Time per Step (ms)")
    ax.set_title("MLP Training Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mlp_training_performance.png")
    plt.close()

    fig, ax = plt.subplots()
    ratio = np.array(results["torch_times"]) / np.array(results["nova_times"])
    ax.plot(results["batch_sizes"], ratio, "o-", color=COLORS[2], linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup (PyTorch / NovaNN)")
    ax.set_title("MLP Training Speedup")
    ax.set_xscale("log")
    ax.legend(["Speedup", "Baseline (1x)"], frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mlp_training_speedup.png")
    plt.close()

    print(f"\n✓ MLP plots saved to {OUTPUT_DIR}")


def plot_convnet_results(results):
    """Plot ConvNet training benchmark results."""

    fig, ax = plt.subplots()
    ax.plot(
        results["batch_sizes"],
        results["nova_times"],
        "o-",
        label="NovaNN",
        color=COLORS[0],
    )
    ax.plot(
        results["batch_sizes"],
        results["torch_times"],
        "s-",
        label="PyTorch",
        color=COLORS[1],
    )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Time per Step (ms)")
    ax.set_title("ConvNet Training Performance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "convnet_training_performance.png")
    plt.close()

    print(f"✓ ConvNet plots saved to {OUTPUT_DIR}")


def plot_optimizer_comparison(results):
    """Plot optimizer comparison results."""

    fig, ax = plt.subplots()
    x_pos = np.arange(len(results["optimizers"]))
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        results["nova_times"],
        width,
        label="NovaNN",
        color=COLORS[0],
        alpha=0.8,
    )
    ax.bar(
        x_pos + width / 2,
        results["torch_times"],
        width,
        label="PyTorch",
        color=COLORS[1],
        alpha=0.8,
    )

    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Time per Step (ms)")
    ax.set_title("Training Performance by Optimizer")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results["optimizers"])
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "optimizer_comparison.png")
    plt.close()

    print(f"✓ Optimizer comparison plots saved to {OUTPUT_DIR}")


def print_summary(mlp_results, conv_results, opt_results):
    """Print comprehensive benchmark summary."""

    print("\n" + "=" * 70)
    print("END-TO-END TRAINING BENCHMARK RESULTS (CPU)")
    print("=" * 70)

    print("\n--- MLP Training (largest batch) ---")
    idx = -1
    batch_size = mlp_results["batch_sizes"][idx]
    print(f"\nBatch Size: {batch_size}")
    print(f"  NovaNN:  {mlp_results['nova_times'][idx]:.4f} ms/step")
    print(f"  PyTorch: {mlp_results['torch_times'][idx]:.4f} ms/step")
    print(
        f"  Ratio:   {mlp_results['torch_times'][idx] / mlp_results['nova_times'][idx]:.2f}x"
    )

    avg_mlp_ratio = np.mean(
        np.array(mlp_results["torch_times"]) / np.array(mlp_results["nova_times"])
    )
    print(f"\nAverage MLP speedup: {avg_mlp_ratio:.2f}x")

    print("\n--- ConvNet Training (largest batch) ---")
    batch_size = conv_results["batch_sizes"][idx]
    print(f"\nBatch Size: {batch_size}")
    print(f"  NovaNN:  {conv_results['nova_times'][idx]:.4f} ms/step")
    print(f"  PyTorch: {conv_results['torch_times'][idx]:.4f} ms/step")
    print(
        f"  Ratio:   {conv_results['torch_times'][idx] / conv_results['nova_times'][idx]:.2f}x"
    )

    avg_conv_ratio = np.mean(
        np.array(conv_results["torch_times"]) / np.array(conv_results["nova_times"])
    )
    print(f"\nAverage ConvNet speedup: {avg_conv_ratio:.2f}x")

    print("\n--- Optimizer Comparison ---")
    for i, opt_name in enumerate(opt_results["optimizers"]):
        ratio = opt_results["torch_times"][i] / opt_results["nova_times"][i]
        print(f"\n{opt_name}:")
        print(f"  NovaNN:  {opt_results['nova_times'][i]:.4f} ms/step")
        print(f"  PyTorch: {opt_results['torch_times'][i]:.4f} ms/step")
        print(f"  Ratio:   {ratio:.2f}x")

    print("\n--- Overall Performance ---")
    all_ratios = (
        list(np.array(mlp_results["torch_times"]) / np.array(mlp_results["nova_times"]))
        + list(
            np.array(conv_results["torch_times"]) / np.array(conv_results["nova_times"])
        )
        + list(
            np.array(opt_results["torch_times"]) / np.array(opt_results["nova_times"])
        )
    )
    overall_avg = np.mean(all_ratios)
    print(f"Average speedup across all benchmarks: {overall_avg:.2f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    nova.manual_seed(42)
    torch.manual_seed(42)

    print("Running end-to-end training benchmarks (CPU)...")
    print("This may take several minutes...\n")

    with Timer() as total_timer:
        print("Benchmarking MLP training...")
        batch_sizes_mlp = [16, 32, 64, 128, 256]
        with Timer() as mlp_timer:
            mlp_results = benchmark_mlp_training(batch_sizes_mlp)
        print(f"\n✓ MLP benchmark completed in {mlp_timer.elapsed:.2f}s")

        print("\nBenchmarking ConvNet training...")
        batch_sizes_conv = [8, 16, 32, 64]
        with Timer() as conv_timer:
            conv_results = benchmark_convnet_training(batch_sizes_conv)
        print(f"\n✓ ConvNet benchmark completed in {conv_timer.elapsed:.2f}s")

        print("\nBenchmarking optimizer comparison...")
        with Timer() as opt_timer:
            opt_results = benchmark_optimizer_comparison(batch_size=64)
        print(f"\n✓ Optimizer benchmark completed in {opt_timer.elapsed:.2f}s")

    print(f"\n✓ Total benchmark time: {total_timer.elapsed:.2f}s")

    print("\nGenerating visualizations...")
    with Timer() as plot_timer:
        plot_mlp_results(mlp_results)
        plot_convnet_results(conv_results)
        plot_optimizer_comparison(opt_results)
    print(f"✓ Plotting completed in {plot_timer.elapsed:.2f}s")

    print_summary(mlp_results, conv_results, opt_results)
