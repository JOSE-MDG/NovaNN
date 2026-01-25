# `nn` Module

The **`nn/`** directory contains **high-level abstractions for building neural networks in NovaNN**.  
This module provides layers, modules, activation functions, losses, parameter initialization, and utilities for building deep learning models in a modular and declarative way.

The design of `nn` closely follows the **PyTorch** philosophy, offering a familiar and expressive API for defining complex architectures through composition of simple modules.

## Example:

```python
import nova
import nova.nn as nn

# Define a simple model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP()
x = nova.randn(32, 784)  # batch of 32 samples
output = model(x)
print(output.shape)  # (32, 10)
```

## Overall Structure

The `nn/` module is organized into:

- **Main files** at the root that define the public API (`Parameter`, `Buffer`, `functional`, `init`)
- **[`modules/`](#modules-submodule)**: implementations of all layers and modules (Linear, Conv, BatchNorm, etc.)
- **[`utils/`](#utils-submodule)**: utilities for gradient clipping and parameter standardization

## Main Files

### `parameter.py`

Defines the base classes for model parameters and buffers:

- **`Parameter`**: Learnable tensor that is updated during training. Has `requires_grad=True` by default.
- **`Buffer`**: Non-learnable tensor that is part of the model state (e.g., running statistics in BatchNorm). Does not require gradients.
- **`UninitializedParameter`** / **`UninitializedBuffer`**: Lazy versions that are automatically materialized on the first forward pass.
- **`UninitializedTensorMixin`**: Common base class for all uninitialized tensors.
- **`is_lazy(param)`**: Helper function to detect if a parameter/buffer is lazy.

`Parameter` and `Buffer` are fundamental to the module system, as they are automatically registered when assigned as attributes of a `Module`.

### Example:

```python
import nova
from nova.nn import Parameter, Buffer, UninitializedParameter, is_lazy

# Create learnable parameter
weight = Parameter(nova.randn(10, 5))
print(weight.requires_grad)  # True

# Create non-learnable buffer
running_mean = Buffer(nova.zeros(10))
print(running_mean.requires_grad)  # False

# Lazy parameter (materialized later)
lazy_param = UninitializedParameter()
print(is_lazy(lazy_param))  # True
materialized = lazy_param.materialize((3, 3))
print(materialized.shape)  # (3, 3)
```

### `init.py`

Contains **weight initialization functions** for neural networks:

**Main methods:**

- **Xavier/Glorot**: `xavier_normal_()`, `xavier_uniform_()`
  - Designed for linear, sigmoid and tanh activations
  - Maintain stable activation and gradient variance
- **Kaiming/He**: `kaiming_normal_()`, `kaiming_uniform_()`
  - Optimized for ReLU and variants
  - Adjust variance according to fan-in/fan-out
- **Basic**: `uniform_()`, `normal_()`, `constant_()`, `zeros_()`, `ones_()`, `random_()`

### Example:

```python
import nova
from nova.nn import Parameter, init

# Xavier initialization for layers with tanh/sigmoid
weight = Parameter(nova.empty((64, 128)))
init.xavier_normal_(weight)

# Kaiming initialization for layers with ReLU
weight_relu = Parameter(nova.empty((128, 256)))
init.kaiming_normal_(weight_relu, nonlinearity='relu')

# Custom initialization
bias = Parameter(nova.empty(64))
init.constant_(bias, 0.0)

# Get fan-in/fan-out information
fan_in, fan_out = init.get_fans(weight, mode='both')
print(f"Fan-in: {fan_in}, Fan-out: {fan_out}")  # Fan-in: 128, Fan-out: 64
```

**Utilities:**

- `calculate_gain(nonlinearity)`: Calculates the recommended gain factor for each activation type
- `get_fans(tensor, mode)`: Calculates fan-in and fan-out from tensor shape

All functions operate **in-place** (suffix `_`) and temporarily disable gradient tracking during initialization.

### `functional.py`

Provides **functional versions** of all neural network operations. This module is analogous to `torch.nn.functional` and allows using layers without maintaining state.

**Function categories:**

#### Activations

- **Rectified**: `relu()`, `leaky_relu()`, `prelu()`
- **Smooth**: `sigmoid()`, `tanh()`, `gelu()`
- **Normalization**: `softmax()`, `log_softmax()`, `normalize()`

### Example:

```python
import nova.nn.functional as F

x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU
print(F.relu(x))  # tensor([0., 0., 0., 1., 2.])

# Softmax
logits = nova.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.5]])
probs = F.softmax(logits, dim=1)
print(probs.sum(dim=1))  # tensor([1., 1.]) (sums to 1 per row)
```

#### Loss Functions

- **Regression**: `mse_loss()`, `l1_loss()`, `smooth_l1_loss()`
- **Binary classification**: `binary_cross_entropy()`, `binary_cross_entropy_with_logits()`
- **Multiclass classification**: `nll_loss()`, `cross_entropy()`
- **Others**: `kl_div()` (KL divergence for distillation and generative models)

All loss functions support:

- Configurable reduction (`'none'`, `'mean'`, `'sum'`, `'batchmean'`)
- Per-element or per-class weights
- Numerical stability through careful implementations

### Example:

```python
import nova.nn.functional as F

# MSE Loss
predictions = nova.tensor([2.5, 0.0, 2.0, 8.0])
targets = nova.tensor([3.0, -0.5, 2.0, 7.0])
loss = F.mse_loss(predictions, targets)
print(loss)  # 0.375

# Cross Entropy
logits = nova.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
targets = nova.tensor([0, 1], dtype=nova.long)
loss = F.cross_entropy(logits, targets)
print(loss)  # scalar loss
```

#### Linear and Convolutional Layers

- **Linear**: `linear()` - affine transformation y = xW^T + b
- **Convolutions**: `conv1d()`, `conv2d()`, `conv3d()`
  - Support for stride, padding, dilation
  - Multiple padding modes ('zeros', 'reflect', 'replicate', 'circular')
  - Efficient implementation via im2col
- **Utilities**: `flatten()` - flattens dimensions for fully connected layers

### Example:

```python
import nova.nn.functional as F

# Conv2D
x = nova.randn(1, 3, 32, 32)  # (batch, channels, height, width)
weight = nova.randn(16, 3, 3, 3)  # (out_channels, in_channels, kh, kw)
output = F.conv2d(x, weight, kernel_size=3, padding=1)
print(output.shape)  # (1, 16, 32, 32)
```

#### Pooling

**Average Pooling:**

- `avg_pool1d()`, `avg_pool2d()`, `avg_pool3d()`
- `global_avg_pool1d()`, `global_avg_pool2d()`, `global_avg_pool3d()`

**Max Pooling:**

- `max_pool1d()`, `max_pool2d()`, `max_pool3d()`
- Support for dilation in pooling windows

### Example:

```python
import nova.nn.functional as F

x = nova.randn(1, 64, 32, 32)

# Max Pooling
max_pooled = F.max_pool2d(x, kernel_size=2, stride=2)
print(max_pooled.shape)  # (1, 64, 16, 16)

# Global Average Pooling
global_pooled = F.global_avg_pool2d(x)
print(global_pooled.shape)  # (1, 64, 1, 1)
```

#### Normalization

- **`batch_norm()`**: Batch normalization
  - Computes batch statistics in training
  - Uses running statistics in eval
  - Updates running mean/var with momentum
- **`layer_norm()`**: Layer normalization
  - Independent of batch size
  - Common in Transformers

#### Regularization

- **`dropout()`**: Standard dropout (randomly zeros elements)
- **`dropout2d()`**: Spatial dropout (zeros entire channels in 2D)
- **`dropout3d()`**: Spatial dropout (zeros entire channels in 3D)

### Example:

```python
import nova.nn.functional as F

# Batch Norm
x = nova.randn(4, 3, 8, 8)
running_mean = nova.zeros(3)
running_var = nova.ones(3)
normalized = F.batch_norm(x, running_mean, running_var, training=True)
print(normalized.shape)  # (4, 3, 8, 8)

# Dropout
x = nova.ones((2, 4))
dropped = F.dropout(x, p=0.5, training=True)
print(dropped)  # some elements at 0, others scaled
```

All dropout functions:

- Only active in training mode
- Automatically scale remaining values by 1/(1-p) to maintain expected sum

### `module.py`

Defines the base class **`Module`**, which is the fundamental abstraction for all neural network components in NovaNN.

**Key features:**

**Automatic registration system:**

- Automatically detects and registers `Parameter`, `Buffer`, and sub-`Module` when assigned as attributes
- Maintains three internal dictionaries: `_parameters`, `_buffers`, `_modules`
- Uses `__setattr__` to intercept assignments

**Metaclass `ModuleMeta`:**

- Automatically registers all `Module` subclasses in the serialization system
- Allows safe deserialization with `weights_only=True`
- Doesn't require explicit `@registry_class` decorators

**Iteration API:**

- `parameters(recurse=True)`: iterates over learnable parameters
- `buffers(recurse=True)`: iterates over non-learnable buffers
- `named_parameters()` / `named_buffers()`: named versions
- `named_modules()`: iterates over the entire model hierarchy

**Training modes:**

- `train(mode=True)`: activates training mode (affects Dropout, BatchNorm, etc.)
- `eval()`: activates evaluation mode
- `_training`: internal flag propagated recursively to submodules

**Serialization:**

- `state_dict()`: exports complete state (parameters + persistent buffers)
- `load_state_dict(state_dict)`: loads state from dictionary

**Representation:**

- `__repr__()`: generates readable representation of model hierarchy
- `extra_repr()`: override method to add custom information

**Forward method:**

- Must be implemented by all subclasses
- Defines forward pass computation
- Automatically invoked when calling `module(x)` thanks to `__call__()`

### Example:

```python
import nova
import nova.nn as nn

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.global_avg_pool2d(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

model = CustomNet()

# Iterate over parameters
for name, param in model.named_parameters():
    print(name, param.shape)

# Change mode
model.train()  # training mode
model.eval()   # evaluation mode

# Serialization
state = model.state_dict()
nova.save(state, 'model.pth')
```

## `modules/` Submodule

Contains all concrete implementations of neural network layers and modules.

### `container.py`

**`Sequential`**: Container that executes modules sequentially.

Features:

- Accepts modules as positional arguments or `OrderedDict`
- Indexing by integer or string
- Support for slicing
- `append()`, `extend()`, `insert()`, `pop()` methods
- Automatic forward chains all modules

### Example:

```python
import nova.nn as nn

# Create Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Use the model
x = nova.randn(32, 784)
output = model(x)
print(output.shape)  # (32, 10)

# Access layers
print(model[0])  # Linear(784, 256)

# Add layers
model.append(nn.Softmax(dim=1))
```

### `activation.py`

Implements activation layers as stateful modules:

- **`ReLU()`**: Rectified Linear Unit
- **`LeakyReLU(negative_slope)`**: ReLU with configurable negative slope
- **`PReLU(num_parameters, init)`**: Parametric ReLU with learnable slope
- **`GELU()`**: Gaussian Error Linear Unit
- **`Sigmoid()`**: Sigmoid function
- **`Tanh()`**: Hyperbolic tangent
- **`Softmax(dim)`**: Exponential normalization

All inherit from `Module` and delegate computation to `functional`.

### Example:

```python
import nova
import nova.nn as nn

# Activations as modules
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.2)
prelu = nn.PReLU(num_parameters=1, init=0.25)
gelu = nn.GELU()

x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(relu(x))  # tensor([0., 0., 0., 1., 2.])
print(leaky_relu(x))  # tensor([-0.4, -0.2, 0., 1., 2.])
print(prelu(x)) # tensor([-0.5, -0.25, 0.0, 1.0, 2.0], requires_grad=True)
print(gelu(x)) # tensor([-0.04540229, -0.158808, 0.0, 0.841192, 1.9545977])
```

### `linear.py`

**`Linear(in_features, out_features, bias=True)`**: Fully connected layer (affine transformation).

**Lazy variant:**

- **`LazyLinear(out_features, bias=True)`**: Automatically infers `in_features` on first forward

Features:

- Kaiming uniform initialization by default
- Parameters: `weight` (out_features, in_features), optional `bias`
- Support for automatic input dimension inference

### Example:

```python
import nova
import nova.nn as nn

# Normal Linear
linear = nn.Linear(10, 5, bias=True)
x = nova.randn(3, 10)
output = linear(x)
print(output.shape)  # (3, 5)

# LazyLinear (automatically infers in_features)
lazy_linear = nn.LazyLinear(5)
x = nova.randn(3, 10)
output = lazy_linear(x)  # Materializes in_features=10
print(output.shape)  # (3, 5)
```

### `conv.py`

Implements 1D, 2D, and 3D convolutional layers:

- **`Conv1d`**, **`Conv2d`**, **`Conv3d`**
- Parameters: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `bias`, `padding_mode`

**Lazy variants:**

- **`LazyConv1d`**, **`LazyConv2d`**, **`LazyConv3d`**
- Base class: `_LazyConvXdMixin`
- Infers `in_channels` on first forward

Features:

- Kaiming uniform initialization
- Support for multiple padding modes
- Efficient implementation via im2col/as_strided

### Example:

```python
import nova
import nova.nn as nn

# Normal Conv2d
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
x = nova.randn(1, 3, 32, 32)
output = conv(x)
print(output.shape)  # (1, 64, 32, 32)

# LazyConv2d (automatically infers in_channels)
lazy_conv = nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1)
x = nova.randn(1, 3, 32, 32)
output = lazy_conv(x)  # Materializes in_channels=3
print(output.shape)  # (1, 64, 32, 32)
```

### `batchnorm.py`

Implements batch normalization in 1D, 2D, and 3D:

- **`BatchNorm1d`**, **`BatchNorm2d`**, **`BatchNorm3d`**
- Base class: **`_BatchNorm`**

**Lazy variants:**

- **`LazyBatchNorm1d`**, **`LazyBatchNorm2d`**, **`LazyBatchNorm3d`**
- Base class: **`_LazyNormBase`**

Features:

- Learnable parameters: `weight` (gamma), `bias` (beta)
- Buffers: `running_mean`, `running_var`, `num_batches_tracked`
- Different behavior in train/eval modes
- Configurable momentum for statistics update

### Example:

```python
import nova
import nova.nn as nn

# BatchNorm2d
bn = nn.BatchNorm2d(num_features=64)
x = nova.randn(4, 64, 32, 32)

# Training mode
bn.train()
output_train = bn(x)  # Uses batch statistics
print(output_train.shape)  # (4, 64, 32, 32)

# Eval mode
bn.eval()
output_eval = bn(x)  # Uses running statistics
print(output_eval.shape)  # (4, 64, 32, 32)

# LazyBatchNorm2d
lazy_bn = nn.LazyBatchNorm2d()
output = lazy_bn(x)  # Materializes num_features=64
```

### `layernorm.py`

**`LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`**: Layer normalization.

Features:

- Normalizes over the last specified dimensions
- Independent of batch size
- Common in Transformer architectures

### Example:

```python
import nova
import nova.nn as nn

# LayerNorm for sequences (e.g., Transformers)
ln = nn.LayerNorm(normalized_shape=(512,))
x = nova.randn(32, 128, 512)  # (batch, seq_len, features)
output = ln(x)
print(output.shape)  # (32, 128, 512)

# LayerNorm for images
ln_2d = nn.LayerNorm(normalized_shape=(3, 32, 32))
x = nova.randn(4, 3, 32, 32)
output = ln_2d(x)
print(output.shape)  # (4, 3, 32, 32)
```

### `pooling.py`

Implements pooling operations:

**Average Pooling:**

- **`AvgPool1d`**, **`AvgPool2d`**, **`AvgPool3d`**
- **`GlobalAvgPool1d`**, **`GlobalAvgPool2d`**, **`GlobalAvgPool3d`**

**Max Pooling:**

- **`MaxPool1d`**, **`MaxPool2d`**, **`MaxPool3d`**

Features:

- Support for kernel_size, stride, padding, dilation
- GlobalAvgPool completely collapses spatial dimensions

### Example:

```python
import nova
import nova.nn as nn

# MaxPool2d
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = nova.randn(1, 64, 32, 32)
output = max_pool(x)
print(output.shape)  # (1, 64, 16, 16)

# GlobalAvgPool2d
global_pool = nn.GlobalAvgPool2d()
x = nova.randn(1, 64, 32, 32)
output = global_pool(x)
print(output.shape)  # (1, 64, 1, 1)
```

### `dropout.py`

Implements regularization through dropout:

- **`Dropout(p=0.5)`**: Standard dropout
- **`Dropout2d(p=0.5)`**: Spatial dropout for 2D feature maps
- **`Dropout3d(p=0.5)`**: Spatial dropout for 3D feature maps

Features:

- Only active in training mode
- Automatic scaling of remaining values
- Dropout2d/3d zero entire channels to preserve spatial correlation

### Example:

```python
import nova
import nova.nn as nn

# Standard dropout
dropout = nn.Dropout(p=0.5)
x = nova.ones(4, 10)

dropout.train()
output_train = dropout(x)  # Some elements at 0
print(output_train)

dropout.eval()
output_eval = dropout(x)  # All elements intact
print(output_eval)

# Dropout2d (for CNNs)
dropout2d = nn.Dropout2d(p=0.3)
x = nova.randn(2, 64, 8, 8)
output = dropout2d(x)  # Zeros entire channels
```

### `flatten.py`

**`Flatten(start_dim=1, end_dim=-1)`**: Flattens a range of dimensions.

Common use: preparing convolutional layer output for fully connected layers.

```python
import nova
import nova.nn as nn

flatten = nn.Flatten(start_dim=1)
x = nova.randn(2, 3, 4, 4)
output = flatten(x)
print(output.shape)  # (2, 48) - flattens all except batch
```

### `loss.py`

Implements loss functions as modules:

**Regression:**

- **`MSELoss(reduction='mean')`**: Mean Squared Error
- **`L1Loss(reduction='mean')`**: Mean Absolute Error
- **`SmoothL1Loss(beta=1.0, reduction='mean')`**: Huber loss

**Classification:**

- **`BCELoss(weight=None, reduction='mean')`**: Binary Cross Entropy
- **`BCEWithLogitsLoss(weight=None, pos_weight=None, reduction='mean')`**: BCE with logits (numerically stable)
- **`NLLLoss(weight=None, reduction='mean')`**: Negative Log Likelihood
- **`CrossEntropyLoss(weight=None)`**: Cross Entropy (combines log_softmax + NLL)
- **`KLDivLoss(log_target=False, reduction='mean')`**: Kullback-Leibler Divergence

Features:

- All inherit from `Module`
- Support configurable reduction
- Per-element or per-class weights
- Delegate computation to `functional`

### Example:

```python
import nova
import nova.nn as nn

# MSELoss
criterion = nn.MSELoss()
predictions = nova.tensor([2.5, 0.0, 2.0, 8.0])
targets = nova.tensor([3.0, -0.5, 2.0, 7.0])
loss = criterion(predictions, targets)
print(loss.item())  # 0.375

# CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
logits = nova.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
targets = nova.tensor([0, 1], dtype=nova.long)
loss = criterion(logits, targets)
print(loss.item())  # scalar loss
```

### `lazy.py`

**`LazyModuleMixin`**: Base class for all lazy module variants.

Features:

- Abstract method `initialize_parameters()` that each subclass must implement
- Handles automatic materialization of uninitialized parameters
- Integrates with `UninitializedParameter` and `UninitializedBuffer`
- Allows building models without knowing input dimensions a priori

## `utils/` Submodule

Contains helper utilities for the `nn` module.

### `clip_grad.py`

Functions for gradient clipping (preventing gradient explosion):

- **`clip_grad_norm_(parameters, max_norm, get_norm=False)`**: Clips global gradient norm
- **`clip_grad_value_(parameters, clip_value)`**: Clips individual gradient values

Common use in training RNNs and Transformers.

### Example:

```python
import nova.nn as nn
from nova.nn.utils import clip_grad_norm_, clip_grad_value_

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Clip global gradient norm
total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Total norm: {total_norm}")

# Clip individual values
clip_grad_value_(model.parameters(), clip_value=0.5)
```

### `tensor_utils.py`

Helper functions for layer parameter standardization:

- **`_single(x)`**: Ensures value is always an int
- **`_pair(x)`**: Converts int to 2-element tuple
- **`_triple(x)`**: Converts int to 3-element tuple
- **`add_padding(input, padding, padding_mode)`**: Adds symmetric padding to 3D/4D/5D tensors according to their dimensionality
- **`calculate_out_size(H, W, kernel_size, padding, stride, dilation)`**: Calculates spatial output dimensions after applying convolutions or pooling, supporting 1D, 2D and 3D operations with or without dilation

These functions facilitate handling parameters like `kernel_size`, `stride`, `padding` and `dilation` which can be specified as ints, tuples, or strings (like "valid"), standardizing them to the format required internally by layers

## Design and Philosophy

NovaNN's `nn` module is designed following these principles:

- **Composition over inheritance**: Complex models are built by composing simple modules
- **Separation of concerns**:
  - `Module` handles state and hierarchy
  - `functional` provides stateless operations
  - `Parameter`/`Buffer` encapsulate learnable/non-learnable data
- **Lazy initialization**: Allows defining architectures without knowing all dimensions a priori
- **Consistency with PyTorch**: Familiar API to ease transition and learning
- **Extensibility**: Easy to add new layers by inheriting from `Module` and following established patterns

## Integration with other modules

The `nn` module integrates closely with:

- **[`autograd/`](../autograd/README.md)**: All operations support automatic differentiation
- **[`optim/`](../optim/README.md)**: Optimizers operate on `model.parameters()`
- **[`serialization/`](../serialization/README.md)**: `state_dict()` and `load_state_dict()` allow saving/loading models
- **[`_internal/`](../_internal/README.md)**: Binding system for low-level operations

---

> For more details on specific operations, consult the source code in `modules/` and `functional.py`.
