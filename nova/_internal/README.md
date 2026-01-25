# `_internal` Module

The **`_internal/`** directory contains internal components that allow NovaNN to **dynamically generate and bind operations to the Tensor class**.  
These files are not part of the stable public API but are essential for the framework to function in a modular and scalable way.

## Main Files

### `_binding.py`

This file is the **core of the dynamic binding system**.  
It is responsible for:

- Loading operation configuration from a YAML file (`native_yaml`).
- Retrieving functions registered in the system (`_OPS_REGISTERED`).
- Generating appropriate methods for each operation:
  - **Dunder methods** (e.g., `__add__`)
  - **Reverse methods** (`__radd__`)
  - **Regular methods** (`add`)
  - **In-place variants** (`add_`, `__iadd__`)
- Attaching these functions dynamically to the `Tensor` class.

This mechanism allows NovaNN to **define new operations or modify existing ones without changing the core implementation** of `Tensor`.

### `_generators.py`

Contains the **method generator functions** that `_binding.py` uses to create methods dynamically.  
Includes:

- `make_forward_func`: generates forward methods (like `__add__` or `__mul__`) for unary or binary operations.
- `make_reverse_func`: generates reverse methods (like `__radd__`) when the operation is called from the right side.
- `make_method`: generates regular methods for direct calls (`tensor.add()`).
- `make_inplace_func`: generates in-place methods (`tensor.add_()`) that modify the original tensor.

These functions are responsible for **validating arguments, automatically converting to tensors when needed, and ensuring compatibility with operations that require or don't require gradients**.

---

> Together, `_internal/` provides the **infrastructure that allows NovaNN to connect the operation system with the Tensor class** in a flexible manner, keeping operation definitions, their registration, and their integration into the public API separate.
