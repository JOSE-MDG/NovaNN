# `optim` Module

The **`optim/`** directory implements **optimizers and learning rate schedulers** for training neural networks in NovaNN.

This module provides modern optimization algorithms that update model parameters based on their gradients, as well as strategies for dynamically adjusting the learning rate during training.

The design closely follows the **PyTorch API**, facilitating transitions between frameworks and providing a consistent, familiar interface.

## General Structure

The `optim/` module is organized into:

- **Optimizers**: parameter update algorithms (`SGD`, `Adam`, `AdamW`, `RMSprop`)
- **Schedulers**: dynamic learning rate adjustment strategies (`StepLR`, `CosineAnnealingLR`, `OneCycleLR`)

All optimizers inherit from the base class **`Optimizer`** defined in [`_interfaces/_optimizer.py`](../_interfaces/README.md), which provides the common structure and state management.

## Optimizers

### `sgd.py`

**`SGD(parameters, lr, momentum=0.0, weight_decay=0.0)`**: Stochastic Gradient Descent with optional momentum.

**Features:**

- **Basic SGD**: Simple update θ = θ - lr \* ∇θ
- **Momentum**: Accumulates velocity to smooth updates and accelerate convergence
  - v = momentum \* v + ∇θ
  - θ = θ - lr \* v
- **Weight decay**: L2 regularization applied to the gradient
- **BatchNorm exception**: Does not apply weight decay to parameters marked with `is_bn_param=True`

**When to use:**

- Simple and robust baseline
- Small or medium-sized datasets
- When fine control over the optimization process is needed
- Convex or nearly convex problems

**Internal state per parameter:**

- `velocity`: momentum accumulator (if momentum > 0)

### `adam.py`

**`Adam(parameters, lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)`**: Standard Adam optimizer with coupled weight decay.

**Features:**

- **Adaptive moments**: Maintains exponential moving averages of the gradient (first moment) and its square (second moment)
  - m = β₁ _ m + (1 - β₁) _ ∇θ
  - v = β₂ _ v + (1 - β₂) _ ∇θ²
- **Bias correction**: Corrects zero-initialization bias
  - m̂ = m / (1 - β₁ᵗ)
  - v̂ = v / (1 - β₂ᵗ)
- **Adaptive update**: Scales learning rate per parameter based on gradient history
  - θ = θ - lr \* m̂ / (√v̂ + ε)
- **Coupled weight decay**: Applied to the gradient before calculating moments

**When to use:**

- General deep learning problems
- Large datasets
- When learning rate is difficult to tune manually
- Training deep networks with noisy gradients

**Internal state per parameter:**

- `step`: step counter (for bias correction)
- `exp_avg`: first moment (moving average of gradient)
- `exp_avg_sq`: second moment (moving average of squared gradient)

### `adamw.py`

**`AdamW(parameters, lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)`**: Adam with decoupled weight decay.

**Features:**

- **Decoupled weight decay**: Unlike standard Adam, weight decay is applied **after** calculating adaptive moments
  - θ = θ - lr _ wd _ θ (decoupled decay)
  - θ = θ - lr \* m̂ / (√v̂ + ε) (Adam update)
- **Better regularization**: Decoupled decay works better with adaptive learning rates
- **Same Adam algorithm**: Identical to Adam except for weight decay order

**When to use:**

- Training Transformers and large models
- When strong regularization is needed
- Transfer learning and fine-tuning
- Superior alternative to Adam with weight decay

**Key difference Adam vs AdamW:**

```
Adam:   grad = grad + wd * param  →  adaptive update
AdamW:  adaptive update  →  param = param - lr * wd * param
```

**Internal state per parameter:**

- `step`: step counter
- `exp_avg`: first moment
- `exp_avg_sq`: second moment

### `rmsprop.py`

**`RMSprop(parameters, lr, alpha=0.99, weight_decay=0.0, momentum=0.0, centered=True, eps=1e-8)`**: Root Mean Square Propagation.

**Features:**

- **Moving average of squared gradients**: Normalizes by the square root of the moving average of squared gradient
  - E[g²] = α _ E[g²] + (1 - α) _ g²
  - θ = θ - lr \* g / (√E[g²] + ε)
- **Centered variant**: Option to normalize by centered variance (more stable)
  - E[g] = α _ E[g] + (1 - α) _ g
  - Var[g] = E[g²] - E[g]²
  - θ = θ - lr \* g / (√Var[g] + ε)
- **Optional momentum**: Can be combined with momentum for better convergence
- **Designed for RNNs**: Originally created for problems with non-stationary gradients

**When to use:**

- Training RNNs and LSTMs
- Problems with highly variable gradients
- Alternative to Adam when simplicity is desired
- Datasets with significant noise

**Internal state per parameter:**

- `step`: step counter
- `exp_avg_sq`: second moment (E[g²])
- `exp_avg`: first moment (E[g], only if centered=True)
- `velocity`: momentum accumulator (if momentum > 0)

## Learning Rate Schedulers

### `lr_scheduler.py`

Contains three strategies for dynamically adjusting the learning rate during training.

#### `StepLR(optimizer, step_size, gamma=1.0, last_epoch=-1)`

Reduces the learning rate multiplicatively every `step_size` epochs.

**Formula:**

```
lr = initial_lr * gamma^(epoch // step_size)
```

**Features:**

- Simple and predictable
- Step decay
- Useful when you know approximately when training stalls

**When to use:**

- Standard CNN training
- When you have a fixed epoch budget
- For fine-tuning after pre-training

#### `CosineAnnealingLR(optimizer, T_max, eta_min=0.0, last_epoch=-1)`

Reduces the learning rate following a cosine curve from `base_lr` to `eta_min`.

**Formula:**

```
lr = eta_min + (base_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
```

**Features:**

- Smooth, continuous decay
- Avoids sharp drops that can destabilize training
- Widely used in ImageNet training

**When to use:**

- Long training of large models
- When smooth convergence is desired
- Deep learning competitions (very popular)
- As part of warm restart cycles

#### `OneCycleLR(optimizer, max_lr, total_steps, pct_start=0.3, div_factor=25.0, final_div_factor=1e4, cycle_momentum=True, max_momentum=0.95, last_epoch=-1)`

Implements the 1cycle policy: learning rate increases linearly to `max_lr`, then decays with cosine annealing.

**Phases:**

1. **Warmup** (pct_start \* total_steps): lr grows from `initial_lr` to `max_lr`
2. **Annealing** (remainder): lr decays with cosine from `max_lr` to `final_lr`

**Features:**

- **Inverse momentum cycle**: Momentum high → low → high (optional)
- **Super-convergence**: Allows training with much higher learning rates
- **Implicit regularization**: The cycle acts as a regularizer
- Based on the "Super-Convergence" paper by Leslie Smith

**When to use:**

- Fast training with fewer epochs
- When you want to maximize learning rate without divergence
- Transfer learning and fine-tuning
- Time-limited training

**Momentum configuration:**

- Automatically detects if optimizer has `momentum` or `betas`
- Adjusts inversely: when lr goes up, momentum goes down (and vice versa)

## Base Class `Optimizer`

All optimizers inherit from **`Optimizer`** (defined in [`_interfaces/_optimizer.py`](../_interfaces/README.md)).

**Common structure:**

- **`param_groups`**: List of dictionaries, each with parameters and their hyperparameters
  - Allows different learning rates per parameter group
- **`state`**: Dictionary mapping parameters to their internal state (moments, velocities, etc.)
- **`defaults`**: Default hyperparameters for all groups

**Main methods:**

- **`step(closure=None)`**: Executes an optimization step
  - Internally calls `_step_impl()` which each optimizer implements
  - Optionally accepts closure to re-evaluate the loss function
- **`zero_grad(set_to_none=False)`**: Clears gradients of all parameters
  - `set_to_none=True`: Frees memory by setting gradients to None
  - `set_to_none=False`: Sets gradients to zero (faster for subsequent backward passes)
- **`add_param_group(param_group)`**: Adds a new parameter group
  - Useful for fine-tuning with different learning rates per layer

## Base Class `_LRScheduler`

All schedulers inherit from **`_LRScheduler`** (defined in [`_interfaces/_lr_scheduler.py`](../_interfaces/README.md)).

**Common structure:**

- **`optimizer`**: Reference to the optimizer being adjusted
- **`base_lrs`**: Initial learning rates of each param_group
- **`last_epoch`**: Epoch/step counter

**Main methods:**

- **`step()`**: Advances the scheduler and updates the optimizer's learning rate
- **`get_lr()`**: Abstract method that each scheduler implements
  - Calculates the new learning rate based on `last_epoch`
- **`get_last_lr()`**: Returns the last applied learning rate

## Design Philosophy

The NovaNN `optim` module is designed following these principles:

- **Separation of concerns**: Optimizers only handle parameter updates, schedulers only handle learning rate adjustment
- **Flexibility**: param_groups system allows granular per-layer configuration
- **Consistency with PyTorch**: Familiar API to facilitate learning and code portability
- **Explicit state**: Each optimizer maintains its internal state clearly and accessibly
- **Weight decay awareness**: Differentiation between coupled weight decay (Adam) and decoupled (AdamW)
- **BatchNorm awareness**: Optimizers detect BatchNorm parameters to avoid applying weight decay

## Integration with Other Modules

The `optim` module integrates closely with:

- **[`nn/`](../nn/README.md)**: Operates on `model.parameters()` to update weights
- **[`autograd/`](../autograd/README.md)**: Reads automatically computed gradients in `.grad`
- **[`_interfaces/`](../_interfaces/README.md)**: Inherits from base classes `Optimizer` and `_LRScheduler`

## Optimizer Comparison

| Optimizer   | Memory | Speed  | Best for                   | Key Hyperparameters |
| ----------- | ------ | ------ | -------------------------- | ------------------- |
| **SGD**     | Low    | Fast   | Convex, baselines          | lr, momentum        |
| **Adam**    | High   | Medium | General deep learning      | lr, betas           |
| **AdamW**   | High   | Medium | Transformers, large models | lr, weight_decay    |
| **RMSprop** | Medium | Medium | RNNs, noisy gradients      | lr, alpha           |

## Usage Examples

### Example 1: Complete training cycle with SGD and StepLR

```python
import nova
import nova.nn as nn
import nova.nn.functional as F
from nova.optim import SGD
from nova.optim.lr_scheduler import StepLR

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, optimizer and scheduler
model = SimpleNet()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Training cycle
num_epochs = 30
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    inputs = nova.randn(32, 784)
    targets = nova.randint(0, 10, (32,))

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update learning rate
    scheduler.step()

    if epoch % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, LR={current_lr:.6f}")

# Evaluation
model.eval()
test_inputs = nova.randn(10, 784)
with nova.no_grad():
    predictions = model(test_inputs)
    print(f"Predictions shape: {predictions.shape}")
```

### Example 2: Transfer learning with AdamW and different learning rates per layer

```python
import nova
import nova.nn as nn
from nova.optim import AdamW
from nova.optim.lr_scheduler import CosineAnnealingLR

# Pretrained model (simulated)
class PretrainedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor "frozen"
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # New classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = PretrainedCNN()

# Configure different learning rates: features lower, classifier higher
optimizer = AdamW([
    {'params': model.features.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters()}
], lr=1e-3, weight_decay=0.01)

scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

# Fine-tuning
num_epochs = 50
for epoch in range(num_epochs):
    model.train()

    # Image batch
    images = nova.randn(16, 3, 32, 32)
    labels = nova.randint(0, 10, (16,))

    # Forward + backward
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        lr_features = optimizer.param_groups[0]['lr']
        lr_classifier = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")
        print(f"  LR features: {lr_features:.8f}, LR classifier: {lr_classifier:.8f}")

print("\nTraining completed!")

```

### Example 3: Fast training with OneCycleLR and gradient clipping

```python
import nova
import nova.nn as nn
import nova.nn.functional as F
from nova.optim import Adam
from nova.optim.lr_scheduler import OneCycleLR
from nova.nn.utils import clip_grad_norm_

# Deeper model
class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

model = DeepNet()

# Configure optimizer and OneCycleLR
total_steps = 1000
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=total_steps,
    pct_start=0.3,
    cycle_momentum=True
)
criterion = nn.CrossEntropyLoss()

# Training with super-convergence
for step in range(total_steps):
    model.train()

    # Simulate mini-batch
    inputs = nova.randn(64, 784)
    targets = nova.randint(0, 10, (64,))

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass with gradient clipping
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients for stability
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0, get_norm=True)

    optimizer.step()
    scheduler.step()

    # Periodic logging
    if step % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        current_momentum = optimizer.param_groups[0]['betas'][0]
        print(f"Step {step}/{total_steps}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  LR: {current_lr:.6f}, Momentum: {current_momentum:.4f}")
        print(f"  Grad norm: {grad_norm:.4f}")

# Final evaluation
model.eval()
with nova.no_grad():
    test_inputs = nova.randn(100, 784)
    test_outputs = model(test_inputs)
    predictions = test_outputs.argmax(dim=1)
    print(f"\nEvaluation: {predictions.shape[0]} samples processed")
```

### Example 4: Comparison of Adam vs AdamW with weight decay

```python
import nova
import nova.nn as nn
from nova.optim import Adam, AdamW

# Simple model for comparison
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Train with Adam (coupled weight decay)
print("=== Training with Adam ===")
model_adam = TinyNet()
optimizer_adam = Adam(model_adam.parameters(), lr=0.01, weight_decay=0.1)

for step in range(5):
    x = nova.randn(8, 10)
    y = nova.randn(8, 1)

    pred = model_adam(x)
    loss = F.mse_loss(pred, y)

    optimizer_adam.zero_grad()
    loss.backward()
    optimizer_adam.step()

    print(f"Step {step}: Loss={loss.item():.4f}, "
          f"Weight norm={nova.norm(model_adam.fc.weight, ord=2).item():.4f}")

# Train with AdamW (decoupled weight decay)
print("\n=== Training with AdamW ===")
model_adamw = TinyNet()
optimizer_adamw = AdamW(model_adamw.parameters(), lr=0.01, weight_decay=0.1)

for step in range(5):
    x = nova.randn(8, 10)
    y = nova.randn(8, 1)

    pred = model_adamw(x)
    loss = F.mse_loss(pred, y)

    optimizer_adamw.zero_grad()
    loss.backward()
    optimizer_adamw.step()

    print(f"Step {step}: Loss={loss.item():.4f}, "
          f"Weight norm={nova.norm(model_adamw.fc.weight, ord=2).item():.4f}")

print("\nNote: AdamW typically produces better regularization in large models")
```

### Example 5: Saving and loading optimizer state

```python
import nova
import nova.nn as nn
from nova.optim import SGD
from nova.optim.lr_scheduler import StepLR

# Create model and optimizer
model = nn.Linear(10, 5)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Train for some steps
for step in range(3):
    x = nova.randn(4, 10)
    y = nova.randn(4, 5)

    pred = model(x)
    loss = F.mse_loss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    print(f"Step {step}: Loss={loss.item():.4f}")

# Save complete checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state,
    'scheduler_last_epoch': scheduler.last_epoch,
    'epoch': 3
}
nova.save(checkpoint, 'checkpoint.pth')

# Create new model and optimizer
new_model = nn.Linear(10, 5)
new_optimizer = SGD(new_model.parameters(), lr=0.1, momentum=0.9)
new_scheduler = StepLR(new_optimizer, step_size=5, gamma=0.5)

# Load checkpoint
loaded = nova.load('checkpoint.pth')
new_model.load_state_dict(loaded['model_state_dict'])
new_optimizer.state = loaded['optimizer_state_dict']
new_scheduler.last_epoch = loaded['scheduler_last_epoch']

# Continue training
for step in range(3, 6):
    x = nova.randn(4, 10)
    y = nova.randn(4, 5)

    pred = new_model(x)
    loss = F.mse_loss(pred, y)

    new_optimizer.zero_grad()
    loss.backward()
    new_optimizer.step()
    new_scheduler.step()

    print(f"Step {step}: Loss={loss.item():.4f}")
```

---

> For more details on specific implementations, consult the source code of each optimizer and scheduler.
