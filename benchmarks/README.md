# Performance Benchmarks

This directory contains scripts and technical reports comparing the performance of **NovaNN** against **PyTorch**. The goal is to measure the efficiency of the autograd engine, memory management, and framework scalability.

## Autograd Backward Overhead

This benchmark measures the additional cost (_overhead_) introduced by the autograd system when computing gradients compared to a forward-only execution.

### Relative Overhead (%)

Shows the stability of the system. While other frameworks may show fluctuations depending on cache state, NovaNN maintains constant overhead.

<p align="center">
  <img src="../images/benchmarks/autograd/relative_overhead.png" width="450" height="450" alt="Relative Overhead">
</p>

### Scalability vs Depth

We measure how the engine scales when graph complexity increases. NovaNN demonstrates linear and predictable growth.

<p align="center">
  <img src="../images/benchmarks/autograd/overhead_vs_depth.png"  width="450" height="450" alt="Overhead vs Depth">
</p>

### Technical Analysis

- **Stability:** NovaNN maintains controlled relative overhead (approx. 50-70%) unlike the initial fluctuations of PyTorch.
- **Predictability:** Autograd latency is consistent, facilitating training time estimation in deep architectures.

**Script:** [`backward_overhead.py`](./autograd/backward_overhead.py)

## Autograd Gradient Accumulation

We compare the efficiency of processing accumulated micro-batches versus a single large batch, a vital technique for training large models on hardware with limited memory.

### Framework Comparison

Direct performance between NovaNN and PyTorch using gradient accumulation strategies.

<p align="center">
  <img src="../images/benchmarks/autograd/accumulation_framework_comparison.png"  width="450" height="450" alt="Framework Comparison">
</p>

### Accumulation Overhead

Percentage cost of splitting batches into micro-steps. NovaNN optimizes synchronization to minimize this impact.

<p align="center">
  <img src="../images/benchmarks/autograd/accumulation_overhead.png"  width="450" height="450" alt="Accumulation Overhead">
</p>

### Technical Analysis

- **Memory Efficiency:** NovaNN optimizes accumulation by reducing internal synchronization overhead between micro-batches.
- **Micro-batching:** The system is robust when splitting workloads, maintaining performance parity with PyTorch but with more controlled memory footprint.

**Script:** [`grad_accumulation.py`](./autograd/grad_accumulation.py)

## Memory Footprint Analysis

We evaluate RAM (RSS) memory consumption during training phases. This benchmark is critical for determining **NovaNN**'s ability to handle deep models or large batches without saturating the system.

### Batch Size Impact

We analyze how memory consumption grows when increasing the amount of data processed simultaneously. A lower slope indicates better management of temporary tensors.

<p align="center">
  <img src="../images/benchmarks/autograd/memory_vs_batch.png" width="450" height="450" alt="Memory vs Batch Size">
</p>

### Graph Overhead (Graph Retention)

We measure retained memory needed to store the computational graph and gradients as network depth increases.

<p align="center">
  <img src="../images/benchmarks/autograd/memory_overhead.png" width="450" height="450" alt="Memory Overhead vs Depth">
</p>

### Technical Analysis

- **Linear Efficiency:** NovaNN demonstrates predictable memory scaling. Unlike frameworks that aggressively reserve large memory cache blocks (like PyTorch's _caching allocator_), NovaNN maintains a usage profile more adjusted to actual demand.
- **Graph Optimization:** Storage of operations for the _backward pass_ remains compact, allowing training of deeper networks on hardware with limited resources.

**Script:** [`memory_footprint.py`](./autograd/memory_footprint.py)

## Element-wise Operations (CPU)

Performance evaluation of element-wise operations on CPU, comparing **NovaNN** versus **PyTorch** on tensors of different sizes (from 10² to 10⁶ elements).

### Key Results

<p align="center">
  <img src="../images/benchmarks/operations/addition_performance.png" width="450" height="450" alt="Element-wise Addition Performance">
</p>

<p align="center">
  <img src="../images/benchmarks/operations/activation_comparison.png" width="450" height="450" alt="Activation Functions Comparison">
</p>

### Technical Analysis

- **Scalability in element-wise addition:** NovaNN shows almost identical behavior to PyTorch on large sizes (10⁶ elements), with only a slight disadvantage in the intermediate range (10⁴–10⁵). Log-log scaling is practically linear in both cases, indicating effective vectorization and low fixed overhead.
- **Activation functions (fixed size = 10,000):** NovaNN shows higher latency in ReLU (~3–4×) and Sigmoid (~2×), while in Tanh the difference is significantly reduced. This points to the fact that implementations of non-linear functions (especially those with exponentials or divisions) still have room for improvement in NovaNN, unlike linear operations which are already highly optimized.
- **Consistency and predictability:** Unlike some frameworks that show latency spikes on small tensors due to dispatch overhead, NovaNN maintains stable and predictable times throughout the evaluated range, which is especially valuable in graphs with many small-sized operations.

**Script:** [`elementwise_cpu.py`](./operations/elementwise_cpu.py)

## Reduction Operations (CPU)

Benchmark of fundamental reduction operations (sum, mean, variance, standard deviation), critical for statistics, normalization layers, loss computation, and metrics during training.

### Key Results

<p align="center">
  <img src="../images/benchmarks/operations/sum_performance.png" width="450" height="450" alt="Sum Reduction Performance">
</p>

<p align="center">
  <img src="../images/benchmarks/operations/statistical_reductions_comparison.png" width="450" height="450" alt="Statistical Reductions Comparison">
</p>

### Technical Analysis

- **Sum reduction:** NovaNN scales practically parallel to PyTorch, reaching parity on large sizes (10⁶ elements). PyTorch's small advantage in the 10⁴–10⁵ range is likely due to better SIMD or cache utilization at that sweet spot, but disappears at larger scales.
- **Statistical reductions (variance and std dev):** NovaNN is ~2–3× slower on a tensor of 10,000 elements. This is expected since these operations require multiple passes (mean calculation + deviation), and the current implementation does not yet include kernel fusion or advanced parallel reduction optimizations like those in ATen/MKL in PyTorch.
- **Practical implications:** The impact is low in most modern architectures where reductions are not the main bottleneck. However, in models with many normalization layers (LayerNorm, GroupNorm, BatchNorm with per-channel statistics) or intensive metric monitoring during training, optimizing these primitives could offer notable CPU performance improvements.

**Script:** [`reduction_ops.py`](./operations/reduction_ops.py)

## End-to-End Training (CPU)

Complete evaluation of end-to-end training performance on CPU, measuring the entire training pipeline: forward pass, loss computation, backward pass, and optimizer step. This benchmark evaluates **NovaNN** versus **PyTorch** in two representative architectures (MLP and ConvNet) and different optimizers.

### MLP Training Performance

We measure training step time of a simple MLP network (3 fully connected layers) varying batch size from 16 to 256 samples.

### Key Results

<p align="center">
  <img src="../images/benchmarks/training/mlp_training_performance.png" width="450" height="450" alt="MLP Training Performance">
</p>

<p align="center">
  <img src="../images/benchmarks/training/mlp_training_speedup.png" width="450" height="450" alt="MLP Training Speedup">
</p>

### Technical Analysis

- **Batch size scalability:** NovaNN scales almost identically to PyTorch, with very similar slope in log-log. For the largest batch (256), NovaNN is only ~1.1× slower, showing that the complete pipeline is already highly optimized.
- **Relative speedup:** NovaNN vs PyTorch speedup reaches a maximum of ~1.1× on small batches (16), but decreases to parity on large batches. This indicates NovaNN has lower fixed overhead, but both frameworks equally exploit parallelization in large matrix operations.
- **Pipeline stability:** Unlike individual operation benchmarks, here we see that NovaNN maintains consistency across the entire batch size range, without abrupt spikes that could indicate memory or internal synchronization issues.

### ConvNet Training Performance

We evaluate training performance of a simple ConvNet (2 convolutional layers + 2 fully connected) with batches from 8 to 64 images of 28×28 (similar to MNIST).

### Key Results

<p align="center">
  <img src="../images/benchmarks/training/convnet_training_performance.png" width="450" height="450" alt="ConvNet Training Performance">
</p>

### Technical Analysis

- **Convolution performance:** NovaNN shows linear scaling similar to PyTorch, reaching parity (~1.0×) on the largest batch (64). PyTorch's slight advantage on small batches is likely due to specific convolution optimizations in ATen/CUDA, although we are on CPU here.
- **Complex architecture overhead:** For ConvNets, which involve more diverse operations (convolutions, pooling, reshaping), NovaNN maintains a minimal difference (<1.2×), demonstrating that the autograd engine and complex graph management are well implemented.
- **Implications for vision:** Results suggest NovaNN is viable for computer vision tasks on CPU, especially in scenarios where memory is limited and smaller batches are preferred.

### Optimizer Comparison

We compare complete pipeline performance using four common optimizers (SGD, Adam, AdamW, RMSprop) with fixed batch size of 64.

### Key Results

<p align="center">
  <img src="../images/benchmarks/training/optimizer_comparison.png" width="450" height="450" alt="Training Performance by Optimizer">
</p>

### Technical Analysis

- **Differences by optimizer:** PyTorch shows significant advantage in Adam (~1.5×), likely due to more mature implementations with operation fusion and better vectorization of adaptive updates. NovaNN is more competitive in SGD (~1.04×), Adam (~1.1x) and RMSprop (~1.2×).
- **Internal state overhead:** Adaptive optimizers (Adam, AdamW) maintain internal states (moments, variances) that require additional memory and extra computations. NovaNN shows higher overhead in these cases, suggesting optimization opportunities in optimizer state management.
- **Practical choice:** For simple training with SGD, NovaNN offers performance almost identical to PyTorch. For adaptive optimizers, the difference is notable but still acceptable for prototyping and experimentation on CPU.

### General Analysis

- **Global performance:** NovaNN achieves an average speedup of ~1.15× vs PyTorch across all benchmarks, with almost perfect parity in fundamental operations and slight disadvantages in complex optimizers.
- **Scalability:** Both architectures (MLP and ConvNet) scale linearly with batch size, indicating good memory management and parallelization in NovaNN.
- **Use cases:** NovaNN is especially competitive for rapid prototyping, CPU training, and scenarios where predictability and low fixed overhead are prioritized over absolute performance.

**Script:** [`end_to_end_cpu.py`](./training/end_to_end_cpu.py)
