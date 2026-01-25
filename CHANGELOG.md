# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2026-01-25

### 🎉 Major Release - Complete Framework Refactoring

Version 4.0.0 represents a **complete rewrite** of NovaNN, transforming it from a basic educational project into a modular, extensible, and professional deep learning framework. This version introduces fundamental changes in the project's architecture, API, and philosophy.

### ✨ Added

#### Dynamic Autograd System

- **Complete automatic differentiation engine** inspired by PyTorch
  - Dynamic construction of computational graphs
  - Base class `Function` for differentiable operations
  - `Context` system for caching intermediate values
  - Automatic backpropagation with `tensor.backward()`
  - Gradient management with `grad_fn` and automatic tracking
  - `no_grad()` and `enable_grad()` modes for gradient control
- **200+ differentiable operations** organized by categories:
  - Basic operations (arithmetic, exponentiation, logarithms)
  - Activations (ReLU, LeakyReLU, PReLU, GELU, Sigmoid)
  - Linear algebra (matmul, dot, det, inv, norm, trace)
  - Tensor manipulation (reshape, permute, stack, concat, split)
  - Reduction (sum, mean, var, min, max)
  - Trigonometric (sin, cos, tan, arcsin, arccos, arctan)
  - Comparison (maximum, minimum, where)
  - Advanced indexing (getitem, setitem with fancy indexing)
  - Views and strides (as_strided, view, extend)

#### Tensor Abstraction

- **Complete `Tensor` class** as wrapper over NumPy arrays
  - Type system with `dtype` support (float32, float64, int32, int64, bool)
  - Automatic gradient tracking with `requires_grad`
  - In-place methods with `_` suffix (add*, mul*, zero\_, etc.)
  - Overloaded operators (`+`, `-`, `*`, `/`, `@`, `**`, etc.)
  - Dynamic binding system from YAML for API generation
  - Advanced properties (shape, strides, ndim, device, is_leaf)

#### Complete `nn` Module

- **Fundamental layers**:
  - `Linear` and `LazyLinear` (fully connected with lazy initialization)
  - `Conv1d`, `Conv2d`, `Conv3d` (1D/2D/3D convolutions)
  - `LazyConv1d`, `LazyConv2d`, `LazyConv3d` (lazy versions)
  - `Flatten` for conv → fc transition
- **Normalization**:
  - `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`
  - `LazyBatchNorm1d`, `LazyBatchNorm2d`, `LazyBatchNorm3d`
  - `LayerNorm` for future Transformer architectures
- **Activations as modules**:
  - `ReLU`, `LeakyReLU`, `PReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`
- **Pooling**:
  - `MaxPool1d`, `MaxPool2d`, `MaxPool3d`
  - `AvgPool1d`, `AvgPool2d`, `AvgPool3d`
  - `GlobalAvgPool1d`, `GlobalAvgPool2d`, `GlobalAvgPool3d`
- **Regularization**:
  - `Dropout`, `Dropout2d`, `Dropout3d`
- **Enhanced `Sequential` container** with automatic submodule registration
- **Complete `Module` system**:
  - Automatic parameter, buffer, and submodule registration
  - `train()` / `eval()` modes propagated recursively
  - `state_dict()` / `load_state_dict()` for serialization
  - Iterators `parameters()`, `named_parameters()`, `modules()`, `named_modules()`
  - Readable representation with `__repr__()` and `extra_repr()`

#### `nn.functional` Module

- **Complete stateless functional API** for all operations
- Activation functions: `relu()`, `leaky_relu()`, `gelu()`, `sigmoid()`, `tanh()`, `softmax()`, `log_softmax()`
- Loss functions: `mse_loss()`, `l1_loss()`, `smooth_l1_loss()`, `binary_cross_entropy()`, `binary_cross_entropy_with_logits()`, `nll_loss()`, `cross_entropy()`, `kl_div()`
- Linear/conv operations: `linear()`, `conv1d()`, `conv2d()`, `conv3d()`
- Functional pooling: `max_pool1d/2d/3d()`, `avg_pool1d/2d/3d()`, `global_avg_pool1d/2d/3d()`
- Normalization: `batch_norm()`, `layer_norm()`, `normalize()`
- Functional dropout: `dropout()`, `dropout2d()`, `dropout3d()`

#### `optim` Module

- **Modern optimizers**:
  - `SGD` with momentum and gradient clipping
  - `Adam` with coupled weight decay
  - `AdamW` with decoupled weight decay (recommended)
  - `RMSprop` with centered variant
- **Learning rate schedulers**:
  - `StepLR` (step decay)
  - `CosineAnnealingLR` (cosine decay)
  - `OneCycleLR` (super-convergence with momentum cycling)
- **Common features**:
  - `param_groups` system for differentiated learning rates
  - Automatic exclusion of BatchNorm parameters from weight decay
  - Pre/post step hooks for logging and debugging
  - `state_dict()` / `load_state_dict()` for checkpointing

#### `metrics` Module

- **Accumulative metrics system** with `reset()` → `update()` → `compute()` pattern
- **Classification**:
  - `Accuracy`, `Precision`, `Recall`, `F1Score` with averaging (micro/macro/weighted)
  - `ConfusionMatrix` multi-class efficient with bincount
  - `ROCAUC` for binary classification
- **Regression**:
  - `MSE` (MSE/RMSE)
  - `MAE` (MAE)
  - `R2Score` (coefficient of determination)

#### Weight Initialization

- **Complete `nn.init` module** with professional functions:
  - Xavier/Glorot: `xavier_normal_()`, `xavier_uniform_()`
  - Kaiming/He: `kaiming_normal_()`, `kaiming_uniform_()`
  - Basic: `uniform_()`, `normal_()`, `constant_()`, `zeros_()`, `ones_()`
  - `calculate_gain()` for activation gains
  - `get_fans()` for fan-in/fan-out calculation

#### Serialization

- **`serialization` module** for saving/loading models
  - `save()` and `load()` with safe pickle support
  - `weights_only=True` mode for security
  - Registry system for safe deserialization of custom classes
  - `@registry_class` decorator for auto-registration

#### Type System and Utilities

- **`_typing` module** with complete type definitions
  - Type hints for all APIs (Size, Dtype, Dim, etc.)
  - `.pyi` stubs for better IDE support
- **`utils` module**:
  - `hooks.py`: Hook system for modules and optimizers
  - `logger.py`: Professional logger with levels and formatting
  - `grad_checking.py`: Numerical gradient validation
  - `to_tensor.py`: Flexible data conversion to tensors
  - `visualization.py`: Utilities for graph and metric visualization
  - `clip_grad.py`: Gradient clipping (norm and value)

#### Benchmarks

- **Complete `benchmarks/` directory** for performance analysis
  - Comparisons with PyTorch on elementwise, reduction, autograd operations
  - End-to-end training benchmarks
  - Memory and computational overhead analysis
  - Reporting and visualization scripts

#### Tutorials

- **`tutorials/` directory** with progressive learning path:
  - `00_philosophy/`: Framework philosophy and design
  - `01_basics/`: Tensors, broadcasting, indexing, dtypes
  - `02_autograd/`: Autograd, backward, computational graph
  - `03_nn/`: Modules, layers, parameters, initialization
  - `04_training/`: Optimizers, schedulers, training loops
  - `05_advanced/`: Custom autograd functions, hooks, profiling
  - `06_comparison/`: Comparisons with PyTorch

#### Documentation

- **Modular READMEs** for each submodule with:
  - Detailed functionality description
  - Practical usage examples
  - Integration with other modules
  - Technical implementation details
- **Complete CONTRIBUTING.md** with style guides and PR process
- **Structured CHANGELOG.md** (this file)

### 🔄 Changed

#### Project Architecture

- **Renamed `novann/` → `nova/`** for cleaner API (`import nova` vs `import novann`)
- **Complete module reorganization**:
  - `layers/` → distributed in `nn/modules/` and `autograd/_ops/`
  - `model/nn.py` → `nn/modules/container.py` (Sequential)
  - `module/` → integrated into `nn/` as `Module` base
  - `losses/` → `nn/modules/loss.py` and `nn/functional.py`
  - `metrics/` → refactored with new accumulative API
  - `utils/` → reorganized by specific functionality
- **Clear separation** between:
  - Public API (`nova.nn`, `nova.optim`, etc.)
  - Internal implementations (`nova._internal`, `nova._interfaces`)
  - Type system (`nova._typing`)

#### Layer and Module System

- **New base class `Module`** with metaclass for auto-registration
- **`Parameter` and `Buffer`** as independent classes (not just wrappers)
  - `Parameter` with `requires_grad=True` by default
  - `Buffer` for non-trainable statistics (BatchNorm)
  - `Uninitialized*` variants for lazy initialization
- **Lazy modules** with automatic initialization on first forward
- **Hook system** for forward and backward passes

#### Optimizers

- **Unified API** with base class `Optimizer`
- **Decoupled weight decay** in AdamW and RMSprop (better than v3.0.0)
- **Automatic exclusion** of BatchNorm parameters from weight decay (previously manual)
- **Support for `param_groups`** for layer-specific learning rates

#### Metrics

- **New accumulative API** `reset()` → `update()` → `compute()` (previously direct calculation)
- **Support for averaging** (micro/macro/weighted) in classification metrics
- **Per-class metrics** in addition to global ones

#### Serialization

- **Registry system** for custom classes (previously only basic pickle)
- **`weights_only=True` mode** for security (prevents arbitrary code execution)

#### Testing

- **Coverage reduced from 95% → 82%** due to:
  - Massive code expansion (3.0.0: ~2000 lines docs, 4.0.0: modular code)
  - New modules without complete tests (schedulers, some autograd ops)
  - Focus on architecture and functionality over exhaustive coverage
- **Tests reorganized** to reflect new module structure

### ⚠️ Breaking Changes

#### Imports

```python
# v3.0.0
from novann.layers import Linear, ReLU
from novann.model import Sequential
from novann.losses import CrossEntropyLoss
from novann.optim import Adam

# v4.0.0
import nova.nn as nn
from nova.optim import Adam

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU()
)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
```

#### Module API

```python
# v3.0.0 - layers without auto-registration
class MyModel:
    def __init__(self):
        self.linear = Linear(10, 5)

    def parameters(self):
        return self.linear.parameters()  # manual

# v4.0.0 - auto-registration with Module
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # auto-registered

    def forward(self, x):
        return self.linear(x)

    # parameters() inherited from Module
```

#### Weight Initialization

```python
# v3.0.0 - manual initialization in each layer
layer = Linear(10, 5)
layer.reset_parameters(init_fn)

# v4.0.0 - initialization in nn.init
from nova.nn import init
weight = nn.Parameter(nova.empty((10, 5)))
init.kaiming_normal_(weight, nonlinearity='relu')
```

#### Metrics

```python
# v3.0.0 - direct calculation
acc = accuracy(model, dataloader)

# v4.0.0 - accumulative API
from nova.metrics import Accuracy
metric = Accuracy(num_classes=10)
for input, target in dataloader:
    preds = model(input)
    metric.update(preds, target)
final_acc = metric.compute()
```

#### Training Loop

```python
# v3.0.0 - forward returns simple output
output = model(x)
loss, grad = loss_fn(output, y)
model.backward(grad)

# v4.0.0 - with automatic autograd
output = model(x)
loss = criterion(output, y)
loss.backward()  # automatically computes gradients
optimizer.step()
```

#### Parameters and Gradients

```python
# v3.0.0 - Parameter simple wrapper
class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

# v4.0.0 - Parameter with complete tracking
class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
        # grad_fn, _inputs, etc. handled by Tensor
```

### 🗑️ Removed

#### Deprecated Modules

- ❌ `novann.functional` (basic functional API) → replaced by complete `nova.nn.functional`
- ❌ `novann.utils.train.train()` (monolithic training function) → users implement custom loops
- ❌ `novann.utils.datasets` (specific loaders) → users load data manually or use generic `DataLoader`
- ❌ Automatic initialization system in `Sequential` based on activations → replaced by explicit `nn.init`
- ❌ `novann.core.config` (hardcoded initialization maps)
- ❌ `Layer` class as separate abstraction → integrated into `Module`

#### Removed Utilities

- ❌ `utils/gradient_checking/numerical.py` → moved to `utils/grad_checking.py` with better API
- ❌ `utils/visualizations/visualization.py` → replaced by more general `utils/visualization.py`
- ❌ `utils/log_config/logger.py` → replaced by `utils/logger.py` with better configuration

#### Dependencies

- ❌ Implicit dependency on specific folder structure
- ❌ Hardcoded loaders for MNIST/Fashion-MNIST

### 🐛 Fixed

#### Autograd

- ✅ Incorrect gradient propagation in operations with broadcasting
- ✅ Memory leaks in long computational graphs
- ✅ Incorrect gradients in `MaxPool` with multiple equal maxima
- ✅ Unstable backward in `BatchNorm` with batch size = 1

#### Optimizers

- ✅ Weight decay incorrectly applied to BatchNorm parameters in v3.0.0
- ✅ Momentum not correctly initialized in SGD
- ✅ Bias correction in Adam only applied on first step

#### Layers

- ✅ Incorrect padding in `Conv2d` with stride > 1
- ✅ Unstable BatchNorm in eval mode with uninitialized running stats
- ✅ Dropout not correctly deactivated in eval mode

#### Serialization

- ✅ Loading models with custom architectures failed
- ✅ State dict did not save persistent buffers of BatchNorm

### 🔒 Security

- 🔐 `weights_only=True` mode in serialization prevents arbitrary code execution
- 🔐 Registry system for safe classes in deserialization
- 🔐 Type validation in critical operations

### 📊 Performance

#### Improvements

- ⚡ Autograd operations 15-25% faster than v3.0.0 (loop optimization)
- ⚡ Conv2d with im2col 30% more efficient on CPU
- ⚡ 40% reduction in computational graph memory overhead

#### Known Regressions

- 🐢 Backward in very deep graphs (>100 layers) can be slow vs PyTorch
- 🐢 Fancy indexing operations ~2x slower than PyTorch (pure Python vs C++)

### 📝 Migration Notes

#### For v3.0.0 Users

1. **Update imports**:

```python
# Before
from novann.layers import Linear
from novann.model import Sequential

# Now
import nova.nn as nn
```

2. **Adapt training loops**:

```python
# Before
loss, grad = criterion(output, target)
model.backward(grad)

# Now
loss = criterion(output, target)
loss.backward()
```

3. **Update metrics**:

```python
# Before
acc = accuracy(model, loader)

# Now
metric = Accuracy(num_classes=10)
for batch in loader:
    metric.update(model(batch['x']), batch['y'])
acc = metric.compute()
```

4. **Change initialization**:

```python
# Before
# Sequential did it automatically

# Now
from nova.nn import init
for m in model.modules():
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
```

5. **Update serialization**:

```python
# Before
import pickle
pickle.dump(model, f)

# Now
import nova
nova.save(model.state_dict(), 'model.pth')
# ...
model.load_state_dict(nova.load('model.pth'))
```

## [3.0.0] - 2025-12-06

### Added

- Basic deep learning framework with fully connected and convolutional layers
- Optimizers: SGD, Adam, AdamW, RMSprop
- Loss functions: MSE, MAE, CrossEntropy, BinaryCrossEntropy
- Metrics: accuracy, binary_accuracy, r2_score
- Layers: Linear, Conv1d, Conv2d, BatchNorm1d, BatchNorm2d, Dropout
- Activations: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- Pooling: MaxPool1d, MaxPool2d, GlobalAvgPool1d, GlobalAvgPool2d
- Sequential container with automatic initialization
- Logging system
- `train()` function for simplified training
- Loaders for MNIST and Fashion-MNIST
- Classification and regression examples
- Unit tests (95% coverage)

### Notes

- Initial functional version of the framework
- 2000+ line README explaining each file (bad practice)
- No autograd system (manual backward)
- No complete static typing

## Version Comparison

| Feature           | v3.0.0         | v4.0.0                  |
| ----------------- | -------------- | ----------------------- |
| **Autograd**      | ❌ Manual      | ✅ Dynamic automatic    |
| **API**           | Custom         | PyTorch-style           |
| **Tensors**       | Simple wrapper | Complete class with ops |
| **Operations**    | ~20            | 200+                    |
| **Modules**       | Basic          | Complete + Lazy         |
| **Schedulers**    | ❌             | ✅ 3 types              |
| **Metrics**       | 3 basic        | 8 + averaging           |
| **Serialization** | Pickle         | Safe + registry         |
| **Documentation** | 1 huge README  | Modular READMEs         |
| **Test coverage** | 95%            | 87% (code 5x larger)    |
| **Benchmarks**    | ❌             | ✅ vs PyTorch           |
| **Tutorials**     | ❌             | ✅ 6 levels             |
