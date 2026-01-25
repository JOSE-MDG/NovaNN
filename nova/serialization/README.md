# `serialization` Module

The **`serialization/`** directory implements functionality for **saving and loading NovaNN objects** in a safe and reproducible way.

This module allows persisting models, weights, optimizer states, and any serializable object, using pickle as the backend but with additional security layers to prevent arbitrary code execution during deserialization.

## General Structure

The module is organized into:

- **`save.py`**: Public function for saving objects
- **`load.py`**: Public function for safely loading objects
- **`_safe_load.py`**: Restricted unpickler for safe deserialization

## Main Files

### `save.py`

Implements the **`save()`** function, which serializes NovaNN objects to disk or buffers.

**Features:**

- **Pickle serialization**: Uses Python's pickle protocol
- **Multi-target support**: Saves to files (str/Path) or buffers (BytesIO)
- **Automatic directory creation**: Creates parent directory if it doesn't exist
- **Logging**: Reports success or errors during saving
- **Error handling**: Captures and reports specific errors (permissions, IO, pickle)

**Signature:**

```python
def save(
    obj: Any,
    f: str | Path | io.BufferedIOBase,
    protocol: int = pickle.HIGHEST_PROTOCOL
) -> None
```

**Parameters:**

- `obj`: Object to serialize (Module, Tensor, state_dict, etc.)
- `f`: File path or buffer to save to
- `protocol`: Pickle protocol version (defaults to latest)

**Usage examples:**

(code block marker 2)

```python
import nova
import nova.nn as nn
from pathlib import Path
import io

# 1. Save complete model
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
nova.save(model, "model.pth")
# ✅ Saved successfully to model.pth

# 2. Save only weights (state_dict)
state = model.state_dict()
nova.save(state, "weights.pth")

# 3. Save with Path object
checkpoint_dir = Path("checkpoints")
nova.save(model, checkpoint_dir / "epoch_10.pth")

# 4. Save to memory buffer
buffer = io.BytesIO()
nova.save(model, buffer)
# Useful for sending over network or storing in database
bytes_data = buffer.getvalue()

# 5. Save complete training state
checkpoint = {
    'epoch': 42,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'loss': 0.123
}
nova.save(checkpoint, "training_checkpoint.pth")

# 6. Use specific pickle protocol
nova.save(model, "model_v4.pth", protocol=4)
```

**Exceptions:**

- `SaveError`: Generic error during serialization
- `PermissionError`: No write permissions
- `TypeError`: Invalid `f` argument

**When to use `save()`:**

- Save trained models for later use
- Create checkpoints during training to resume later
- Export weights for transfer learning
- Persist experimental configurations

### `load.py`

Implements the **`load()`** function, which deserializes objects **safely by default**.

**Main features:**

- **Safe loading by default**: `weights_only=True` prevents arbitrary code execution
- **Restricted unpickler**: Only allows explicitly registered classes
- **Optional unsafe fallback**: `weights_only=False` for compatibility (not recommended)
- **Multi-source support**: Loads from files or buffers
- **Path validation**: Verifies existence before attempting to load
- **Detailed logging**: Reports success, warnings, and errors

**Signature:**

```python
def load(
    f: str | Path | io.BufferedIOBase,
    *,
    weights_only: bool = True
) -> Any
```

**Parameters:**

- `f`: File path or buffer to load from
- `weights_only`: If True, uses safe unpickler (recommended). If False, uses standard pickle (security risk)

**Usage examples:**

(code block marker 4)

```python
import nova
from pathlib import Path
import io

# 1. Load saved model (safe by default)
model = nova.load("model.pth")
# ✅ Successfully loaded from model.pth

# 2. Load state_dict and apply to model
new_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
state = nova.load("weights.pth")
new_model.load_state_dict(state)

# 3. Load with Path object
checkpoint_path = Path("checkpoints/epoch_10.pth")
model = nova.load(checkpoint_path)

# 4. Load from memory buffer
buffer = io.BytesIO(saved_bytes)
buffer.seek(0)  # Important: return to start
model = nova.load(buffer)

# 5. Load complete training checkpoint
checkpoint = nova.load("training_checkpoint.pth")
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
scheduler.load_state_dict(checkpoint['scheduler_state'])
start_epoch = checkpoint['epoch'] + 1
print(f"Resuming from epoch {start_epoch}")

# 6. Unsafe loading (NOT RECOMMENDED)
# Only use if you completely trust the source
model = nova.load("model.pth", weights_only=False)
# ⚠️ Warning: Loading with weights_only=False is unsafe
```

**Exceptions:**

- `FileNotFoundError`: File doesn't exist
- `LoadError`: Generic error during deserialization
- `UnsafeLoadError`: Attempt to load unregistered class with `weights_only=True`
- `TypeError`: Invalid `f` argument

**When to use `weights_only=True` (default):**

- Load weights from external or untrusted sources
- Production environments where security is critical
- When you only need to load state_dicts or registered modules

**When to use `weights_only=False` (security risk):**

- Load arbitrary Python objects you completely trust
- Debugging or development where you control the source
- Compatibility with old checkpoints containing unregistered classes

**NEVER use `weights_only=False` with files from unknown or untrusted sources.**

### `_safe_load.py`

Implements **`SafeUnpickler`**, a restricted unpickler that prevents arbitrary code execution.

**Purpose:**

Deserialization with pickle can execute malicious code if a file contains instructions to instantiate arbitrary classes. `SafeUnpickler` solves this by implementing allowlists of permitted modules and classes.

**Allowlists:**

**Allowed modules:**

```python
ALLOWED_MODULES = {
    "numpy",
    "numpy.core.multiarray",
    "numpy.core.numeric",
    "numpy._core.numeric",
    "numpy._core.multiarray",
    "nova.dtypes",
}
```

**Allowed built-in types:**

```python
ALLOWED_BUILTINS = {
    "dict", "list", "tuple", "set", "frozenset",
    "int", "float", "str", "bytes", "bool",
    "complex", "bytearray", "range", "slice",
    "type", "object", "NoneType"
}
```

**Registered NovaNN classes:**

Any class decorated with `@registry_class` is automatically allowed. This includes:

- Tensors
- All `nn` modules (Linear, Conv2d, ReLU, etc.)
- Optimizers (SGD, Adam, AdamW, RMSprop)
- Schedulers (StepLR, CosineAnnealingLR, OneCycleLR)
- Autograd functions
- Metrics

**Key method:**

```python
def find_class(self, module_name: str, global_name: str)
```

This method is called during deserialization to resolve each class. It implements the allowlist logic:

1. Checks if module is in `ALLOWED_MODULES`
2. Allows specific NumPy internals needed for arrays
3. Allows safe built-in types
4. Allows `OrderedDict` (used in state_dicts)
5. Searches in NovaNN class registry (`@registry_class`)
6. **Blocks everything else** and raises `UnsafeLoadError`

**Example of registered class:**

```python
from nova.utils import registry_class

@registry_class
class MyCustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(nova.randn(10, 10))

    def forward(self, x):
        return x @ self.weight

# Now MyCustomLayer can be safely loaded
model = MyCustomLayer()
nova.save(model, "custom.pth")

# This works because MyCustomLayer is registered
loaded = nova.load("custom.pth", weights_only=True)  # ✅ OK
```

**Example of unregistered class:**

```python
# Without @registry_class
class UnsafeLayer:
    def __init__(self):
        super().__init__()
        self.data = [1, 2, 3]

model = UnsafeLayer()
nova.save(model, "unsafe.pth")

# This fails because UnsafeLayer is not registered
try:
    loaded = nova.load("unsafe.pth", weights_only=True)
except UnsafeLoadError as e:
    print(e)
    # Blocked unpickling of unregistered class: __main__.UnsafeLayer
    # To fix this, either:
    #   1. Register the class using @registry_class decorator
    #   2. Load with weights_only=False (not recommended)
```

**Helper function:**

```python
def _load_from_file(file: io.BufferedIOBase, weights_only: bool = True) -> Any
```

Internal helper used by `load()` that decides between `SafeUnpickler` and standard `pickle.load()` based on `weights_only`.

## Complete Serialization Flow

### Saving (save):

1. User calls `nova.save(obj, "model.pth")`
2. Validates that `obj` is not None
3. Creates parent directory if it doesn't exist
4. Serializes with `pickle.dump(obj, file, protocol)`
5. Logs success

### Safe loading (load with weights_only=True):

1. User calls `nova.load("model.pth")`
2. Verifies file exists
3. Opens file in binary mode
4. Creates instance of `SafeUnpickler(file)`
5. During `unpickler.load()`:
   - Each class is validated against allowlists
   - Only NumPy, builtins, and registered classes are allowed
   - Arbitrary classes are blocked
6. Returns the deserialized object

### Unsafe loading (load with weights_only=False):

1. User calls `nova.load("model.pth", weights_only=False)`
2. Shows security warning in logs
3. Uses standard `pickle.load()` (no restrictions)
4. **Any class can be deserialized** (security risk)

## Integration with Other Modules

The `serialization` module integrates with:

- **[`nn/`](../nn/README.md)**: All modules support `state_dict()` / `load_state_dict()`
- **[`optim/`](../optim/README.md)**: Optimizers and schedulers are serializable
- **[`utils/decorators/registry.py`](../utils/README.md)**: The `@registry_class` decorator registers classes for safe loading
- **`_tensor.py`**: Tensors are directly serializable

## Design Philosophy

The `serialization` module follows these principles:

- **Security by default**: `weights_only=True` prevents deserialization attacks
- **Explicit opt-in for risks**: `weights_only=False` requires conscious user decision
- **Explicit registration**: Only classes decorated with `@registry_class` are trusted
- **Compatibility**: Supports both paths and buffers for flexibility
- **Clear logging**: Descriptively reports successes, warnings, and errors
- **Separation of concerns**: `save` doesn't know security details, `_safe_load` handles that

## Advanced Examples

### Example 1: Periodic checkpointing during training

```python
import nova
import nova.nn as nn
from nova.optim import Adam
from nova.optim.lr_scheduler import CosineAnnealingLR

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

best_loss = float('inf')

for epoch in range(100):
    # ... training loop ...

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'loss': current_loss,
        }
        nova.save(checkpoint, f"checkpoints/epoch_{epoch}.pth")

    # Save best model
    if current_loss < best_loss:
        best_loss = current_loss
        nova.save(model.state_dict(), "best_model.pth")

print("Training completed!")
```

### Example 2: Resume training from checkpoint

```python
import nova
import nova.nn as nn
from nova.optim import Adam
from nova.optim.lr_scheduler import CosineAnnealingLR

# Create model and optimizer
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Load checkpoint
checkpoint = nova.load("checkpoints/epoch_40.pth")
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
scheduler.load_state_dict(checkpoint['scheduler_state'])
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['loss']

print(f"Resuming training from epoch {start_epoch}")

# Continue training
for epoch in range(start_epoch, 100):
    # ... training loop ...
    pass
```

### Example 3: Transfer learning (partial weight loading)

```python
import nova
import nova.nn as nn

# Pretrained model on ImageNet (simplified)
pretrained = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.Linear(128, 1000)  # 1000 ImageNet classes
)

# Save pretrained model
nova.save(pretrained.state_dict(), "imagenet_weights.pth")

# New model for 10 classes (your dataset)
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.Linear(128, 10)  # Only 10 classes
)

# Load partial weights (only convolutional layers)
pretrained_weights = nova.load("imagenet_weights.pth")
model_weights = model.state_dict()

# Copy only compatible layers
for name, param in pretrained_weights.items():
    if name in model_weights and param.shape == model_weights[name].shape:
        model_weights[name] = param

model.load_state_dict(model_weights)
print("Transferred convolutional layers from pretrained model!")
```

### Example 4: Send model over network

```python
import nova
import nova.nn as nn
import io
import socket

# Server: serialize and send
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
buffer = io.BytesIO()
nova.save(model, buffer)
model_bytes = buffer.getvalue()

# Send via socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 12345))
sock.sendall(model_bytes)
sock.close()

# Client: receive and deserialize
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('localhost', 12345))
sock.listen(1)
conn, addr = sock.accept()
received_bytes = conn.recv(4096)

buffer = io.BytesIO(received_bytes)
buffer.seek(0)
model = nova.load(buffer)
print("Model received and loaded!")
```

### Example 5: Register custom class for safe loading

```python
from nova.utils import registry_class
import nova.nn as nn

# Define custom class with registration
@registry_class # Actually not necessary as it registers automatically when inheriting from nn.Module
class CustomAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = (Q @ K.T) / (Q.shape[-1] ** 0.5)
        return attn @ V

# Create and save model with custom class
model = nn.Sequential(
    CustomAttention(512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
nova.save(model, "custom_model.pth")

# Load safely (works because CustomAttention is registered)
loaded_model = nova.load("custom_model.pth", weights_only=True)  # ✅ OK
print("Custom model loaded safely!")
```

---

> For more details about the class registration system, consult [`utils/decorators/registry.py`](../utils/README.md).
