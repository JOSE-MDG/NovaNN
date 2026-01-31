# `utils` Module

The **`utils/`** directory provides **general utilities and helper tools** that support the entire NovaNN framework.

This module contains cross-cutting functionalities that don't belong to any specific category but are essential for the operation, debugging, logging, data handling, and extensibility of the framework.

## Overall Structure

The module is organized into:

- **Main files** with general-purpose utilities
- **[`decorators/`](#decorators-submodule)**: Decorators for registration, timing, and other functionalities
- **[`datasets/`](#datasets-submodule)**: Loaders and utilities for common datasets (MNIST, Fashion-MNIST)
- **[`data/`](#data-submodule)**: Base classes and utilities for data handling (Dataset, DataLoader, preprocessing)

## Main Files

### `logger.py`

Implements a **singleton logging system** for NovaNN with multi-level support and multiple outputs.

**Features:**

- **Singleton Pattern**: A single logger instance across the entire application
- **Multi-output**: Log to console and file simultaneously
- **Configurable levels**: DEBUG, INFO, WARNING, ERROR
- **Customizable format**: Timestamps, levels, function names
- **Thread-safe**: Safe for concurrent use

**`LoggerLevel` Class:**

Enum with available logging levels:

```python
class LoggerLevel(Enum):
    DEBUG = logging.DEBUG      # Detailed debugging information
    INFO = logging.INFO        # General information
    WARNING = logging.WARNING  # Warnings
    ERROR = logging.ERROR      # Errors
```

**`Logger` Class:**

Singleton logger with methods for each level:

**Main methods:**

- `info(msg, **kwargs)`: INFO level log
- `debug(msg, **kwargs)`: DEBUG level log
- `warning(msg, **kwargs)`: WARNING level log
- `error(msg, **kwargs)`: ERROR level log with automatic traceback
- `set_level(level)`: Change level dynamically

**Usage examples:**

```python
from nova.utils.logger import logger, LoggerLevel

# Basic logging
logger.info("Model training started")
logger.debug("Batch size: 32, Learning rate: 0.001")
logger.warning("Gradient norm is high")
logger.error("Failed to load checkpoint")

# With additional kwargs
logger.info("Epoch completed", epoch=10, loss=0.123, acc=0.95)
# Output: ... | INFO | Epoch completed | epoch: 10, loss: 0.123, acc: 0.95

# Change level dynamically
logger.set_level(LoggerLevel.WARNING)  # Only shows WARNING and ERROR
```

**When to use:**

- Training progress tracking
- Framework operation debugging
- Error and warning logging
- Critical operation auditing

### `memory.py`

Provides **`MemoryTracker`** and utilities for memory usage profiling during code execution.

**Features:**

- **Context manager** for automatic memory tracking
- **Baseline adjustment**: Subtracts baseline memory for accurate measurements
- **Peak and current memory**: Tracks maximum and current usage
- **Internal snapshots**: Allows top allocation analysis
- **Multiple units**: Properties for MB and KB
- **Garbage collection**: Automatic cleanup before measuring

**`MemoryTracker` Class:**

Context manager that tracks memory usage using `tracemalloc`.

**Attributes:**

- `verbose`: Whether to automatically print statistics on exit
- `baseline`: Baseline memory in bytes (pre-tracking)
- `peak`: Peak memory in bytes (adjusted for baseline)
- `current`: Current memory in bytes (adjusted for baseline)

**Methods:**

- `__enter__()`: Starts tracking with GC and baseline
- `__exit__()`: Stops tracking, calculates stats, optional verbose print
- `get_top_stats(limit=10)`: Returns top N allocations
- Properties: `peak_mb`, `current_mb`, `peak_kb`, `current_kb`

**Examples:**

```python
from nova.utils.memory import MemoryTracker

# Basic usage
with MemoryTracker() as mem:
    data = [i for i in range(1000000)]
print(f"Peak: {mem.peak_mb:.2f} MB")

# Verbose mode (auto-print)
with MemoryTracker(verbose=True) as mem:
    model = create_large_model()
# ==================================================
# Memory Usage Statistics
# ==================================================
# Peak memory:         125.43 MB
# Current memory:       98.21 MB
# ==================================================

# Analyze top allocations
with MemoryTracker() as mem:
    data = process_large_dataset()
top_stats = mem.get_top_stats(5)
for stat in top_stats:
    print(stat)
```

**Helper functions:**

#### `quick_memory_check(func, *args, **kwargs)`

Executes a function while tracking memory, returns stats + result.

**Returns:**

Dict with keys: `peak_mb`, `current_mb`, `peak_kb`, `current_kb`, `result`

**Examples:**

```python
from nova.utils.memory import quick_memory_check

def create_list(n):
    return [i for i in range(n)]

stats = quick_memory_check(create_list, 1000000)
print(f"Peak: {stats['peak_mb']:.2f} MB")
print(f"Result length: {len(stats['result'])}")
```

#### `compare_memory(nova_func, torch_func, *args, verbose=True, **kwargs)`

Compares memory usage between two methods.

**Returns:**

Tuple `(nova_peak_mb, torch_peak_mb, ratio)`

**Examples:**

```python
from nova.utils.memory import compare_memory

input_peak, torch_peak, ratio = compare_memory(
    input_forward,
    other_forward,
    x_nova, x_torch,
    verbose=True
)
# ==================================================
# Memory Comparison: NovaNN vs PyTorch
# ==================================================
# input_forward peak:       125.43 MB
# other_forward peak:       98.21 MB
# Ratio:              1.28x
# ==================================================
```

**When to use:**

- Memory footprint profiling of expensive operations
- Memory leak debugging
- Benchmarking vs PyTorch
- Memory usage optimization

### `memory_usage.py`

Provides **`@measure_memory`**, a decorator to measure function memory usage.

**`@measure_memory` Decorator:**

**Features:**

- **Dual-mode**: With or without parameters
- **Verbose optional**: Auto-print statistics
- **Return memory optional**: Returns (result, (peak_mb, current_mb))
- **Wraps preserves metadata**: Uses `@wraps` internally

**Parameters:**

- `func`: Function to decorate (automatic)
- `verbose`: Whether to print stats (default: False)
- `return_memory`: Whether to return tuple with stats (default: False)

**Examples:**

```python
from nova.utils.decorators import measure_memory

# Basic usage (no parameters)
@measure_memory
def my_function():
    data = [i**2 for i in range(1000000)]
    return data

result = my_function()  # Only returns result

# With verbose
@measure_memory(verbose=True)
def train_model():
    model = create_model()
    train(model)
    return model

# With return_memory
@measure_memory(return_memory=True, verbose=False)
def compute():
    return heavy_computation()

result, (peak_mb, current_mb) = compute()
print(f"Used {peak_mb:.2f} MB peak")
```

**When to use:**

- Decorating training functions for tracking
- Debugging functions suspected of memory leaks
- Automatic profiling without modifying internal code

### `hooks.py`

Defines **`HooksHandle`**, a handler for safely registering and removing hooks.

**Features:**

- **Simplified management**: Hook registration and removal
- **Duplicate prevention**: Internal `_removed` flag prevents multiple removals
- **Tensor and Optimizer integration**: Used internally by backward hooks and step hooks

**`HooksHandle` Class:**

**Attributes:**

- `hooks_list`: List where the hook is registered
- `hooks_func`: Hook function
- `_removed`: Flag indicating if already removed

**Methods:**

- `remove()`: Removes the hook from the list

**Usage examples:**

```python
import nova

# Usage with Tensor backward hooks
x = nova.tensor([1.0, 2.0], requires_grad=True)

def my_hook(grad):
    print(f"Gradient: {grad}")
    return grad * 2

# Register hook
handle = x.register_hook(my_hook)

# Use the tensor normally
y = (x ** 2).sum()
y.backward()  # Hook executes here

# Remove hook when no longer needed
handle.remove()

# Now the hook no longer executes
y.backward()
```

**When to use:**

- When implementing custom backward hooks
- For temporary gradient debugging
- In systems that need dynamic hooks

### `to_tensor.py`

Implements **`ensure_tensor()`**, a utility function for robust conversion to Tensors.

**Features:**

- **Automatic conversion**: Converts arrays, scalars, lists to Tensors
- **Conditional preservation**: If already a Tensor with no changes, returns the original
- **Dtype inference**: Infers appropriate types based on input
- **Error handling**: Detailed exception logging

**Signature:**

```python
def ensure_tensor(
    obj: Any,
    dtype: Optional[Dtype] = None,
    requires_grad: Optional[bool] = None
) -> Tensor
```

**Use cases:**

**Case 1: Already a Tensor**

```python
t = nova.tensor([1.0, 2.0])
result = ensure_tensor(t)  # Returns the same object
assert result is t
```

**Case 2: NumPy Array**

```python
arr = np.array([1.0, 2.0, 3.0])
t = ensure_tensor(arr, dtype=nova.float32, requires_grad=True)
# Converts to Tensor with specified dtype and requires_grad
```

**Case 3: Python Scalars**

```python
# Automatic dtype inference
ensure_tensor(5)          # dtype=nova.long (int)
ensure_tensor(5.0)        # dtype=nova.float32 (float)
ensure_tensor(True)       # dtype=nova.bool (bool)
ensure_tensor([1, 2, 3])  # dtype=nova.float32 (list of float32)
```

**Case 4: Property override**

```python
t = nova.tensor([1.0], requires_grad=False)
new_t = ensure_tensor(t, requires_grad=True)
# Creates new Tensor with requires_grad=True
```

**When to use:**

- In functions that accept flexible inputs (Tensor, array, scalar)
- To normalize inputs in framework operations
- When safe conversion with fallback is needed

**Used internally in:**

- `nova.nn.functional` (all functions normalize inputs)
- Creation functions
- Autograd operations

### `grad_checking.py`

Provides **`grad_check_wrt_inputs()`** for numerical gradient verification.

**Purpose:**

Compares analytic gradients (calculated by backprop) with numerical gradients (finite differences) to detect bugs in custom operation implementations.

**Signature:**

```python
def grad_check_wrt_inputs(
    fn: Callable[[Tensor], Tensor],
    *args: Tensor,
    eps: float = 1e-4,
    zero_grads: bool = True,
    domain_bounds: Optional[tuple[float, float]] = None,
    **kwargs
) -> tuple[list[ndarray], list[ndarray]]
```

**Parameters:**

- `fn`: Function to verify (must return Tensor)
- `*args`: Input tensors with `requires_grad=True`
- `eps`: Perturbation for finite differences
- `zero_grads`: Whether to clear gradients afterward
- `domain_bounds`: Limits for clamping (e.g., (0, 1) for probabilities)

**Returns:**

- Tuple of `(analytic_grads, numerical_grads)` (lists of arrays)

**Central finite differences method:**

```text
∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)
```

**Usage examples:**

**Example 1: Verify quadratic operation**

```python
from nova.utils import grad_check_wrt_inputs

x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)

def square_sum(t):
    return (t ** 2).sum()

analytic, numeric = grad_check_wrt_inputs(square_sum, x)

# Compare
diff = np.abs(analytic[0] - numeric[0])
print(f"Max difference: {diff.max()}")  # Should be very small (~1e-5)

# Verification with allclose
assert nova.allclose(analytic[0], numeric[0], rtol=1e-3, atol=1e-5)
```

**Example 2: Verify sigmoid function**

```python
def sigmoid_sum(t):
    return nova.sigmoid(t).sum()

x = nova.tensor([0.5, -0.5, 1.0], requires_grad=True)
analytic, numeric = grad_check_wrt_inputs(
    sigmoid_sum, x,
    domain_bounds=(0, 1)  # Clamping for numerical stability
)

assert nova.allclose(analytic[0], numeric[0], rtol=1e-3)
```

**Example 3: Verify custom operation**

```python
from nova.autograd.function import Function

class MyOp(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 3

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out * 3 * (x ** 2)  # Correct?

# Verify
x = nova.tensor([2.0], requires_grad=True)
analytic, numeric = grad_check_wrt_inputs(lambda t: MyOp.apply(t).sum(), x)

if not nova.allclose(analytic[0], numeric[0], rtol=1e-3):
    print("BUG DETECTED: Gradients don't match!")
```

**When to use:**

- When implementing new operations in `autograd/_ops`
- For debugging incorrect gradients
- Testing custom layers or modules
- Validation of mathematical implementations

**Limitations:**

- Slow for large tensors (O(N) complexity in elements)
- Can have numerical issues with discontinuous functions
- Requires differentiable functions

## `decorators/` Submodule

Contains reusable decorators for cross-cutting functionality.

### `registry.py`

Implements the **class registration system** for safe serialization.

**Main decorators:**

#### `@registry_class`

Registers a class for safe deserialization with `nova.load()`.

**Features:**

- **Automatic registration**: Uses module + name as unique key
- **Idempotent**: Re-registering the same class doesn't cause error
- **Used by Module**: All `Module` subclasses automatically registered via metaclass

**Usage example:**

```python
from nova.utils import registry_class
import nova.nn as nn

@registry_class
class CustomLayer:
    def __init__(self, features):
        super().__init__()
        self.weight = nn.Parameter(nova.randn(features, features))

    def forward(self, x):
        return x @ self.weight

# Now CustomLayer can be safely saved and loaded
model = CustomLayer(10)
nova.save(model, "custom.pth")
loaded = nova.load("custom.pth", weights_only=True)  # ✅ OK
```

**When to use:**

- When defining custom modules that will be saved
- For custom optimizers or schedulers
- Any class that will be serialized

#### `@registry_op(op_name)`

Registers an autograd operation for dynamic binding.

**Features:**

- Associates a public name (e.g., "add") with a `Function` class
- Used internally by the binding system
- Validates that only `Function` subclasses are registered
- **Prevents duplicates**: Only registers if name doesn't already exist

**Parameters:**

- `op_name`: Public operation name (e.g., "add", "relu")

**Returns:**

Decorator that registers the `Function` class

**Raises:**

- `ValueError`: If attempting to register something that's not a `Function` subclass

**Examples:**

```python
from nova.utils import registry_op
from nova.autograd.function import Function

@registry_op("custom_op")
class CustomOp(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y + x

    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensors
        return grad_out * (y + 1), grad_out * x

# Now "custom_op" is registered
# And if incorporated in the yaml, it can be used by the binding system
```

**Helper functions:**

- `get_registered_classes(module, name)`: Retrieves class registered by (module, name)
- `_MODULES`: Global dict `{(module, name): class}` with registered classes
- `_OPS_REGISTERED`: Global dict `{op_name: Function}` with registered operations

**When to use:**

- When implementing custom autograd operations
- For operations that will be used in the binding system
- When safe deserialization of computational graphs is needed

### `timing.py`

Provides utilities for measuring execution time, including **`@chronometer`** and **`benchmark`**.

#### `@chronometer`

Decorator to measure function execution time with benchmarking support.

**Features:**

- **Smart formatting**: Adjusts units based on duration (ns, μs, ms, s, m, h)
- **Automatic logging**: Uses NovaNN's logging system
- **Non-invasive**: Returns original result without modification (unless `return_time=True`)
- **Descriptive emojis**: ⚡ (fast), ⏱️ (medium), 🐢 (slow)
- **Benchmarking mode**: Multiple iterations with warmup
- **Dual return mode**: Can return only result or (result, avg_time)

**Parameters:**

- `func`: Function to decorate (automatic)
- `n_iters`: Number of iterations to average (default: 1)
- `warmup`: Warmup iterations not counted (default: 0)
- `return_time`: If True, returns `(result, avg_time)` (default: False)
- `verbose`: If True, logs the time (default: True)

**Returns:**

- If `return_time=False`: function result
- If `return_time=True`: `(result, average_time_in_seconds)`

**Examples:**

```python
from nova.utils.decorators import chronometer

# Basic usage (no parameters)
@chronometer
def train_step(model, batch):
    loss = model(batch)
    loss.backward()
    return loss

# ⚡ train_step: 234ms

# With benchmarking
@chronometer(n_iters=50, warmup=10)
def forward_pass(model, x):
    return model(x)

# ⚡ forward_pass: 12.34ms (avg over 50 runs)

# With return_time (no verbose)
@chronometer(return_time=True, verbose=False)
def compute():
    return expensive_operation()

result, elapsed = compute()
print(f"Took {elapsed*1000:.2f}ms")

# Silent benchmarking
@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def benchmark_op():
    return matrix_multiply(A, B)

result, avg_time = benchmark_op()
```

**Format ranges:**

- `< 1μs`: nanoseconds (ns)
- `< 1ms`: microseconds (μs)
- `< 1s`: milliseconds (ms)
- `< 1min`: seconds (s)
- `< 1h`: minutes + seconds (Xm Ys)
- `≥ 1h`: hours + minutes + seconds (Xh Ym Zs)

**When to use:**

- Profiling training functions
- Debugging bottlenecks
- Benchmarking operations with multiple iterations
- Performance tracking with programmatic time access

#### `benchmark(func, *args, n_iters=100, warmup=10, **kwargs)`

Function to execute precise benchmarks on any callable.

**Description:**

Executes a function multiple times, skipping warmup iterations, and returns:

- The function result
- The average execution time
- The time standard deviation

**Parameters:**

- `func`: Function to benchmark
- `*args`: Positional arguments for the function
- `n_iters`: Number of measured iterations (default: 100)
- `warmup`: Warmup iterations (default: 10)
- `**kwargs`: Keyword arguments for the function

**Returns:**

Tuple `(result, mean_time, std_time)` where:

- `result`: Value returned by the function (last iteration)
- `mean_time`: Average time in seconds (float)
- `std_time`: Standard deviation in seconds (float)

**Examples:**

```python
from nova.utils import benchmark

def matmul(A, B):
    return A @ B

# Basic benchmark
result, mean, std = benchmark(matmul, A, B, n_iters=100)
print(f"Average: {mean*1000:.3f} ms ± {std*1000:.3f} ms")

# With custom warmup
result, mean, std = benchmark(
    neural_net_forward,
    model, x,
    n_iters=50,
    warmup=5
)

# Implementation comparison
nova_res, nova_mean, nova_std = benchmark(nova_conv, x, w)
torch_res, torch_mean, torch_std = benchmark(torch_conv, x, w)
speedup = torch_mean / nova_mean
print(f"Speedup: {speedup:.2f}x")
```

**Features:**

- **Configurable warmup** to stabilize measurements
- **Average and standard deviation** with NumPy for statistical analysis
- **Returns result** in addition to stats for correctness verification
- Ideal for **reproducible performance comparisons**

**When to use:**

- Comparisons with PyTorch implementations
- Evaluation of internal optimizations
- Reproducible measurement in `benchmarks/` scripts
- When you need detailed stats (mean + std) instead of just timing

## `data/` Submodule

Contains abstractions for dataset and dataloader handling.

### `dataset.py`

Defines the abstract base class **`Dataset`**.

**`Dataset` Class:**

Contract for all datasets in NovaNN.

**Abstract methods:**

- `__len__()`: Returns total number of samples
- `__getitem__(index)`: Returns sample(s) at index

**`Index` type:**

```python
type Index = slice | int | tuple | Tensor | ndarray
```

Supports multiple indexing types:

- Integer: `dataset[0]` → single sample
- Slice: `dataset[0:10]` → batch
- List/tuple: `dataset[[1,5,9]]` → fancy indexing
- Tensor: `dataset[nova.tensor([0,2,4])]`
- Array: `dataset[np.array([1,3,5])]`

**Implementation example:**

```python
from nova.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Usage
dataset = MyDataset(nova.randn(100, 10), nova.randint(0, 2, (100,)))
print(len(dataset))  # 100
x, y = dataset[0]    # First sample
batch_x, batch_y = dataset[0:32]  # First batch
```

**When to inherit:**

- When creating custom datasets
- For image, text, audio datasets, etc.
- When lazy loading or augmentation is needed

### `dataloader.py`

Implements **`DataLoader`**, an iterator that yields batches from a Dataset.

**Features:**

- **Automatic batching**: Splits dataset into batches of specified size
- **Configurable shuffling**: Shuffles indices at the start of each epoch
- **Variable last batch**: Automatically handles smaller last batch
- **Efficient iteration**: Doesn't load entire dataset into memory

**`DataLoader` Class:**

**Parameters:**

- `dataset`: Dataset instance
- `batch_size`: Batch size (default: 64)
- `shuffle`: Whether to shuffle indices (default: True)

**Methods:**

- `__iter__()`: Returns iterator for one epoch
- `__len__()`: Returns number of batches
- `batch_size` (property): Read-only access to batch size

**Internal `_Iter` Class:**

Iterator that maintains state for a complete epoch.

**Usage examples:**

**Example 1: Basic training loop**

```python
from nova.utils.data import DataLoader

# Create dataset and loader
dataset = MyDataset(nova.randn(1000, 784), nova.randint(0, 10, (1000,)))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
model.train()
for epoch in range(10):
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
```

**Example 2: Evaluation without shuffling**

```python
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
total_correct = 0
with nova.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        total_correct += (pred.argmax(dim=1) == yb).sum()

accuracy = total_correct / len(test_dataset)
```

**Example 3: Multiple epochs with independent shuffling**

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(5):
    # Each epoch has different shuffling
    for batch_idx, (xb, yb) in enumerate(loader):
        # training...
        pass
    print(f"Epoch {epoch}: {len(loader)} batches")
```

**When to use:**

- In all training/evaluation loops
- To efficiently iterate over large datasets
- When automatic batching and shuffling is needed

### `preprocessing.py`

Preprocessing utilities for data normalization, splitting, saving, and downloading datasets.

**Functions**:

- `normalize(x_data, x_mean, x_std)`: Normalizes data using mean and standard deviation. Supports NumPy arrays and Nova Tensors. Includes an epsilon guard (`1e-8`) to avoid division by zero.

- `split_features_and_labels(df, label_column, dtype)`: Splits a DataFrame into feature and label arrays. If `label_column` doesn't exist, the first column is used as labels. Features default to `float32`, labels are always `int64`.

- `split_validation_subset(x, y, factor, shuffle, stratify, random_state)`:
  Splits arrays or Tensors into train and validation subsets. If inputs are Nova Tensors, they are converted internally and returned as Tensors. Raises `ValueError` if `factor` is not in `(0, 1)`.

- `split_validation_dataset(dataset, label, factor, root, save_method, ...)`: Splits a DataFrame into train and validation sets and saves them to disk. Supports `csv`, `parquet`, and `excel` formats.

- `save_to_csv(df, root)` / `save_to_parquet(df, root)` / `save_to_excel(df, root)`: Save a DataFrame to the specified format. Validates the DataFrame before writing, creates directories if needed, and cleans up partial files on failure.

- `download_dataset(dataset, root, format, force_redownload, validate)`: Downloads MNIST or Fashion-MNIST from their official servers and converts them to a tabular format. Each image is flattened to 784 pixel columns. Already converted files are skipped unless `force_redownload=True`.

**Usage examples:**

**Example 1: normalize data**

```python
from nova.utils.data import normalize
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x_normalized = normalize(x, x_mean=np.mean(x), x_std=np.std(x))
```

**Example 2: split features and labels**

```python
from nova.utils.data import split_features_and_labels
import pandas as pd

df = pd.DataFrame({'label': [0, 1], 'pixel0': [128, 255], 'pixel1': [64, 32]})
x, y = split_features_and_labels(df)
# x.shape -> (2, 2), dtype float32
# y.shape -> (2,),   dtype int64
```

**Example 3: Split validation subset**

```python
from nova.utils.data import split_validation_subset
import numpy as np

x = np.random.rand(1000, 784)
y = np.random.randint(0, 10, 1000)

x_train, y_train, x_val, y_val = split_validation_subset(
    x, y, factor=0.2, stratify=True, random_state=8
)
# x_train.shape -> (800, 784)
# x_val.shape   -> (200, 784)
```

**Example 1: Split and save validation dataset**

```python
from nova.utils.data import split_validation_dataset
import pandas as pd

df = pd.read_parquet("data/mnist_train.parquet")

train, val = split_validation_dataset(
    df,
    label="label",
    factor=0.16,
    root="data/Mnist",
    save_method="parquet",
    set_name="mnist_train_e",
    val_name="mnist_validation",
    random_state=8,
    stratify=True,
)
```

**Example 1: Save DtaFrame**

```python
from nova.utils.data import save_to_csv, save_to_parquet, save_to_excel
import pandas as pd

df = pd.DataFrame({'label': [0, 1], 'pixel0': [128, 255]})

save_to_csv(df, root="output/data.csv")
save_to_parquet(df, root="output/data.parquet")
save_to_excel(df, root="output/data.xlsx")
```

**Example 1: Download datasets from web-sites**

```python
from nova.utils.data import download_dataset

# Download as parquet (recommended)
download_dataset("mnist", root="~/.novann/datasets", format="parquet")

# Force redownload
download_dataset("fashion-mnist", root="~/.novann/datasets", format="parquet", force_redownload=True)
```

## `datasets/` Submodule

Pre-configured loaders for common datasets.

### `mnist.py` and `fashion.py`

Provide functions to load **MNIST** and **Fashion-MNIST** from `.parquet`.

**Functions:**

- `load_mnist_data(...)`: Loads MNIST
- `load_mnist_defatul()`: Loads MNIST with default arguments
- `load_fashion_mnist_data(...)`: Loads Fashion-MNIST
- `load_fashion_mnist_data()`: Loads Fashion-MNIST with default arguments

**Common parameters:**

- `tensor4d`: If True, reshapes to (N, 1, 28, 28) for CNNs
- `as_tensor`: If True, converts to nova.Tensor
- `do_normalize`: If True, normalizes using training set statistics
- `dtype`: Data type for features
- `train_path`, `test_path`, `val_path`: Paths to save. Defaults `~/.novann/datasets`.

**Returns:**

Tuple of 3 datasets: `(train, test, val)`, each is an instance of `MnistData`/`FashionData` (subclasses of `Dataset`).

**Usage examples:**

**Example 1: Load MNIST for MLP**

```python
from nova.utils.datasets import mnist

train, test, val = mnist.load_mnist_data(
    tensor4d=False,  # (N, 784) for MLP
    as_tensor=True,
    do_normalize=True,
    dtype=nova.float32
)

print(len(train))  # ~15000
print(train[0][0].shape)  # (784,)
```

**Example 2: Load Fashion-MNIST for CNN**

```python
from nova.utils.datasets import fashion

train, test, val = fashion.load_fashion_mnist_data(
    tensor4d=True,  # (N, 1, 28, 28) for CNN
    as_tensor=True,
    do_normalize=True,
    dtype=nova.float32
)

print(train[0][0].shape)  # (1, 28, 28)
```

**Example 3: Complete pipeline**

```python
from nova.utils.datasets import mnist
from nova.utils.data import DataLoader

# Load data
train, test, val = mnist.load_mnist_data(dtype=nova.float32)

# Create loaders
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=128, shuffle=False)

# Training loop
for epoch in range(10):
    for xb, yb in train_loader:
        # xb: (64, 784), yb: (64,)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Integration with other modules

The `utils` module integrates with:

- **[`serialization/`](../serialization/README.md)**: `registry_class` registers classes for safe loading
- **[`autograd/`](../autograd/README.md)**: `registry_op` registers operations, `grad_checking` verifies gradients
- **[`nn/`](../nn/README.md)**: `Dataset` and `DataLoader` are fundamental for training
- **Entire framework**: `logger` is used globally, `ensure_tensor` normalizes inputs

## Design and Philosophy

The `utils` module follows these principles:

- **Cross-cutting utilities**: Functionality benefiting multiple modules
- **Minimal dependency**: Utils don't depend on complex components
- **Extensibility**: Decorators and base classes facilitate extension
- **Robustness**: Error handling and detailed logging
- **Performance**: Decorators like `@chronometer` for profiling

---

> For more details on specific components, consult the source code in the corresponding files.
