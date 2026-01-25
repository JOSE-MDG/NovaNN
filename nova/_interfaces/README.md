# `_interfaces` Module

The **`_interfaces/`** directory defines the **abstract base classes** that establish contracts and shared behaviors for key framework components.

These interfaces are not meant to be instantiated directly, but to be inherited by concrete implementations, ensuring API consistency and facilitating framework extensibility.

## Purpose

Interfaces in NovaNN serve several roles:

- **Behavioral contracts**: Define which methods each component must implement
- **Static typing**: Improve experience in IDEs and analysis tools
- **Code reuse**: Implement shared logic among subclasses
- **Living documentation**: Serve as specification of the expected API

## Main Files

### `_optimizer.py`

Defines the base class **`Optimizer`**, which is the contract for all optimizers in NovaNN.

**Responsibilities:**

- **Parameter groups management**: Allows applying different hyperparameters to distinct parameter subsets
- **Optimizer state**: Maintains momentum, adaptive statistics, etc.
- **Hook system**: Supports pre-step and post-step hooks for logging, gradient clipping, etc.
- **Serialization**: Saves and loads the complete optimizer state

**Main attributes:**

- `param_groups`: List of parameter groups, each with its hyperparameters
- `state`: Dictionary mapping parameters to their state (momentum, velocity, etc.)
- `defaults`: Default hyperparameters (lr, weight_decay, etc.)

**Key methods:**

- `_step_impl(closure)`: **Abstract method** that must be implemented by subclasses. Defines the specific update rule of the optimizer.
- `step(closure)`: Executes a complete optimization step (hooks → update → hooks)
- `zero_grad(set_to_none)`: Clears gradients of all parameters
- `add_param_group(group)`: Adds a new parameter group with specific hyperparameters
- `state_dict()` / `load_state_dict()`: State serialization

**Usage example (inheritance):**

```python
from nova._interfaces._optimizer import Optimizer

class MyOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, custom_param=0.5):
        defaults = {'lr': lr, 'custom_param': custom_param}
        super().__init__(params, defaults)

    def _step_impl(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            custom = group['custom_param']

            for param in group['params']:
                if param.grad is None:
                    continue

                # Custom update rule
                param.data -= lr * param.grad * custom

        return loss
```

**When to inherit from `Optimizer`:**

- When implementing a new optimization algorithm (Adagrad, Adadelta, etc.)
- When automatic parameter group management is needed
- To leverage the hook system and serialization

### `_lr_scheduler.py`

Defines the base class **`_LRScheduler`**, which is the contract for all learning rate schedulers.

**Responsibilities:**

- **Dynamic LR adjustment**: Modifies learning rate according to a predefined schedule
- **Optimizer synchronization**: Directly updates the optimizer's `param_groups`
- **State tracking**: Keeps record of the current epoch/step
- **Serialization**: Allows saving and restoring scheduler state

**Main attributes:**

- `optimizer`: Reference to the optimizer whose LR will be adjusted
- `last_epoch`: Index of the last executed epoch/step
- `base_lrs`: Initial learning rates for each parameter group

**Key methods:**

- `get_lr()`: **Abstract method** that must return a list of learning rates (one per parameter group)
- `step()`: Advances the scheduler one step and updates the optimizer's LRs
- `get_last_lr()`: Returns the last applied learning rates
- `state_dict()` / `load_state_dict()`: State serialization

**Usage example (inheritance):**

```python
from nova._interfaces._lr_scheduler import _LRScheduler

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Linear decay from base_lr to 0
        progress = self.last_epoch / self.total_steps
        factor = 1.0 - progress
        return [base_lr * factor for base_lr in self.base_lrs]

# Usage
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = LinearDecayLR(optimizer, total_steps=100)

for epoch in range(100):
    train(...)
    scheduler.step()  # LR decreases linearly
```

**When to inherit from `_LRScheduler`:**

- When implementing a new learning rate adjustment strategy
- When automatic synchronization with the optimizer is needed
- To leverage state serialization

**Schedulers implemented in NovaNN:**

NovaNN includes three concrete schedulers in [`optim/lr_scheduler.py`](../optim/README.md):

- **StepLR**: Decays LR every N epochs by a gamma factor
- **CosineAnnealingLR**: Decays LR following a cosine curve
- **OneCycleLR**: Implements the "1cycle" method (increases and then decreases LR)

### `_base_tensor.py`

Defines the base class **`TensorBase`**, which provides fundamental properties and metadata for all tensors.

**Purpose:**

- Expose a consistent interface for accessing tensor attributes
- Separate metadata logic from mathematical operations
- Provide read-only and computed properties

**Main properties:**

- `data`: Underlying NumPy array (getter/setter with validation)
- `shape`: Tensor shape (tuple of dimensions)
- `dtype`: Data type of elements
- `ndim` / `dim()`: Number of dimensions
- `strides`: Strides in bytes of each dimension
- `T`: Transpose (alias of `permute()`)
- `is_leaf`: Whether it's a leaf node in the computational graph
- `device`: Always returns `'cpu'` (NovaNN doesn't yet support GPU)
- `is_cuda`: Always returns `False`

**Main methods:**

- `size(dim)`: Returns the complete shape or the size of a specific dimension
- `numel()`: Total number of elements
- `itemsize`: Size in bytes of each element
- `nbytes`: Total bytes consumed

**Features:**

- Uses `__slots__ = []` for memory efficiency
- The `data` setter handles automatic conversion of arrays and tensors
- Supports negative indexing in `size(dim)`

**Implementation example:**

```python
import nova

x = nova.randn(3, 4, 5)

# TensorBase properties
print(x.shape)      # (3, 4, 5)
print(x.dtype)      # float32
print(x.dim())       # 3
print(x.numel())    # 60
print(x.size(0))    # 3
print(x.size(-1))   # 5
print(x.strides)    # Strides in bytes
print(x.T.shape)    # (5, 4, 3) - transpose
```

**When to inherit from `TensorBase`:**

- When implementing a new tensor class with custom behavior
- To maintain consistency with the tensor property API
- Generally not directly inherited by end users

## Integration with other modules

The `_interfaces` module integrates with:

- **[`optim/`](../optim/README.md)**: All optimizers (SGD, Adam, AdamW, RMSprop) inherit from `Optimizer`
- **[`optim/lr_scheduler.py`](../optim/README.md)**: All schedulers (StepLR, CosineAnnealingLR, OneCycleLR) inherit from `_LRScheduler`
- **`_tensor.py`**: The `Tensor` class uses `TensorBase` as base for properties

## Design and philosophy

The `_interfaces` module follows these principles:

- **Clear contracts**: Abstract methods (`_step_impl`, `get_lr`) explicitly define what must be implemented
- **Shared functionality**: Methods like `step()`, `zero_grad()`, `state_dict()` are implemented once
- **Extensibility**: Adding new optimizers/schedulers only requires implementing the abstract method
- **Strong typing**: Uses type hints and stubs (`.pyi`) for better IDE experience
- **Automatic registration**: Uses `@registry_class` for safe serialization

---

> For concrete implementations of these interfaces, see [`optim/`](../optim/README.md) for optimizers and schedulers.
