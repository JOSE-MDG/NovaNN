# `_typing` Module

The **`_typing/`** directory contains definitions of **types, protocols, and data structures** used throughout NovaNN.  
These definitions help **correctly type tensors, operations, optimizers, schedulers, datasets, and modules**, improving type safety and autocompletion in editors.

## Main Types

- **Core Tensor Types**: `Size`, `Dtype`, `Dim`, `TensorOrArray`, `Inputs`, `Gradients`.  
  Represent shapes, data types, flexible inputs for tensor operations, and gradient tuples.

- **Autograd Types**: `Hook`, `StepHook`, `Hooks`, `HooksList`, `Closure`.  
  Facilitate typing of gradients, backward hooks, optimizer hooks, and closures.

- **Dataset Types**: `Mnist`, `Fashion`.  
  Types to represent datasets loaded by NovaNN utilities.

- **Module and Parameter Types**: `Modules`, `ModuleTypes`.  
  Represent any module-like object, parameter, or buffer within the framework.

- **Convolution and Pooling Types**: `KernelSize`, `Stride`, `Dilation`, `Padding`, `PaddingMode`.  
  Types for convolution and pooling operations in 1D, 2D, or 3D.

- **Optimizer Types**: `Defaults`, `Group`, `ParamGroups`, `State`, `OptimizerStateDict`.  
  Types that type optimizers, parameter groups, and serializable states.

- **Scheduler Types**: `SchedulerStateDict`.  
  Types for serializing learning rate scheduler states.

- **Loss Function Types**: `LossReduction`.  
  Types that define loss function reduction modes.

- **Metrics Types**: `Average`.  
  Types for metric averaging strategies (micro, macro, weighted).

- **YAML Configuration Types**: `InplaceInfo`, `TensorInfo`, `OperationInfo`, `YAMLFile`.  
  Types describing the structure of the native operations YAML file, including in-place and dunder methods.

- **Binding Types**: `UnaryMethod`, `BinaryMethod`, `ReverseBinaryMethod`, `VariadicMethod`, `InplaceUnaryMethod`, `InplaceBinaryMethod`, `InplaceVariadicMethod`.  
  Protocols defining the signature of dynamically generated methods for tensors, whether unary, binary, reverse, or in-place.

## Detailed Type Descriptions

### Core Tensor Types

- `Size`: Tuple of integers representing tensor shape
- `Dtype`: Union of all supported numeric data types (uint8, int32, float32, etc.)
- `Dim`: Single dimension or tuple of dimensions for operations
- `TensorOrArray`: Flexible input accepting tensors, numpy arrays, or lists/tuples of tensors
- `Inputs`: General input type for operations (tensors, scalars, or any value)
- `Gradients`: Tuple of gradient arrays or None values from backward passes

### Autograd Types

- `Hook`: Backward hook function receiving and optionally modifying gradients
- `StepHook`: Optimizer hook called after parameter updates
- `Hooks`: Union of all hook types
- `HooksList`: List containing any type of hooks
- `Closure`: Optional closure function for optimizers that re-evaluate models

### Dataset Types

- `Mnist`: Return type for MNIST dataset loading function (train, test, validation)
- `Fashion`: Return type for Fashion-MNIST dataset loading function

### Module and Parameter Types

- `Modules`: Union of all module-like objects (Tensor, Optimizer, Parameter, Buffer, Module, etc.)
- `ModuleTypes`: Type objects for module-like classes

### Convolution and Pooling Types

- `KernelSize`: Integer or tuple for 1D, 2D, or 3D kernel sizes
- `Stride`: Optional stride specification for operations
- `Dilation`: Dilation rate for dilated convolutions
- `Padding`: Padding specification (integer, tuple, or 'valid'/'same' modes)
- `PaddingMode`: Padding fill mode ('zeros', 'reflect', 'replicate', 'circular')

### Optimizer Types

- `Defaults`: Default hyperparameter dictionary
- `Group`: Parameter group with associated hyperparameters
- `ParamGroups`: List of parameter groups
- `State`: Optimizer state dictionary mapping parameters to their state
- `OptimizerStateDict`: Complete optimizer state for serialization

### Scheduler Types

- `SchedulerStateDict`: Learning rate scheduler serialization state

### Loss Function Types

- `LossReduction`: Reduction modes for loss functions ('none', 'mean', 'sum', 'batchmean')

### Metrics Types

- `Average`: Averaging strategies for metrics ('micro', 'macro', 'weighted', None)

### YAML Configuration Types

- `InplaceInfo`: Configuration for in-place operation variants
- `TensorInfo`: Configuration for binding operations to Tensor class
- `OperationInfo`: Complete operation definition for YAML configuration
- `YAMLFile`: Root structure of native_functions.yaml

### Binding Types

- `UnaryMethod`: Protocol for unary tensor methods (`__neg__`, `__abs__`, `relu()`)
- `BinaryMethod`: Protocol for binary tensor methods (`__add__`, `__mul__`)
- `ReverseBinaryMethod`: Protocol for reverse binary methods (`__radd__`, `__rmul__`)
- `VariadicMethod`: Protocol for variadic methods (`sum(dim=...)`, `reshape(...)`)
- `InplaceUnaryMethod`: Protocol for in-place unary methods (`abs_()`, `relu_()`)
- `InplaceBinaryMethod`: Protocol for in-place binary methods (`add_()`, `mul_()`)
- `InplaceVariadicMethod`: Protocol for in-place variadic methods (`clamp_()`)

---

> `_typing/` provides the **typing infrastructure and contracts** that ensure consistency between operations, Tensor methods, optimizers, modules, and datasets, without affecting the public API. All types are designed to work seamlessly with Python's static type checkers and IDE autocompletion.
