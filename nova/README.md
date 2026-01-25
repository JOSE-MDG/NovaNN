# `nova` module

The **`nova/`** directory contains the **core of the NovaNN framework** .
Here are implemented all the fundamental abstractions necessary to build, train and analyze deep learning models, as well as the internal infrastructure that allows the system to be extensible, consistent and efficient.

This module defines both the **public API** used by users and the **internal mechanisms** that make the framework work.

## General structure

The `nova/` module is organized in a modular way, clearly separating responsibilities between:

- data representation
- automatic differentiation
- layers and models
- optimization
- metrics
- serialization
- internal utilities

Each submodule has its own detailed documentation.

## Main submodules

- **[`autograd/`](./autograd/README.md)**
  Implement NovaNN's automatic differentiation system.
  It includes the construction of the computational graph, the definition of differentiable functions, the calculation of gradients, and the control of the gradient mode.
- **[`nn/`](./nn/README.md)**
  It contains high-level abstractions for neural networks: modules, layers, activation functions, losses, and training-related utilities.
- **[`optim/`](./optim/README.md)**
  Implements optimizers and learning rate planners used during model training.
- **[`metrics/`](./metrics/README.md)**
  It provides metrics for classification and regression tasks, designed to integrate naturally with `Tensor` .
- **[`serialization/`](./serialization/README.md)**
  It allows you to save and load models, tensors, and training states safely and reproducibly.
- **[`core/`](./core/README.md)**
  Defines global configurations, constants, and base parameters used throughout the framework.
- **[`utils/`](./utils/README.md)**
  It includes general utilities such as registers, hooks, validations, logging, and auxiliary tools.

## Internal files and modules ( `_` )

NovaNN uses the prefix `_` to indicate **internal components** that are not part of the stable public API.

These modules exist to **support the internal architecture of the framework** and are not intended to be used directly by the end user.

### `_internal/`

- **[`_internal/`](./_internal/README.md)**
  It contains the infrastructure that allows NovaNN to dynamically generate and connect operations to the tensor system.
  This module acts as the **internal assembly engine** of the framework and is key to keeping the definition of operations, their registration, and their integration into the public API separate.

### `_interfaces/`

- **[`_interfaces/`](./_interfaces/README.md)**
  Defines base contracts and abstractions for components such as optimizers and learning rate schedulers.
  It facilitates consistency between implementations and static typing.

### `_typing/`

- **[`_typing/`](./_typing/README.md)**
  It provides definitions of auxiliary types and annotations used throughout the project, to improve the experience in editors and static analysis tools.

## Examples of using the NovaNN API

### Create a tensor and perform basic operations

```python
import nova

# Create tensors
x = nova.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = nova.tensor([[2.0, 0.0], [1.0, 3.0]])

# Operations
z = x * y + x
loss = z.sum()
loss.backward()

print(x.grad)  # Autograd

# array([[3. 1.]
#        [2. 4.]])
```

### Define a simple neural network

```python
import nova.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNet()
print(model)

# SimpleNet(
#   (fc1): Linear(in_features=2, out_features=4, bias=True)
#   (relu): ReLU()
#   (fc2): Linear(in_features=4, out_features=1, bias=True)
# )
```

### Define optimizer and basic training

```python
import nova.nn.functional as F
from nova import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss={loss.item()}")

```

### Saving and loading a model

```python
import nova

nova.save(model.state_dict(), "mimodelo.pt")
loaded_model.load_state_dict(nova.load("mimodelo.pt"))
```

### Use of metrics

```python
from nova.metrics import Accuracy

metric = Accuracy(num_classes=2)
metric.reset()

preds = loaded_model(x)
labels = nova.tensor([[1.0], [0.0]])
metric.update(pred, labels)
acc = metric.comput().item()
print("Accuracy:", acc)
```

---

> For more details about each component, consult the specific documentation for each submodule.
