---
name: python-docstrings
description: 'Prompt and workflow for generating and auditing Python docstrings in Google style. Covers modules, classes, functions/methods, properties, dataclasses, and exceptions. Use whenever the user asks to document, write, review, complete, or standardize docstrings in .py files, even if they just say "document this file" or "docstrings are missing."'
---

### Instructions

This file defines how to write and audit Python docstrings using **Google style**, applied consistently to any `.py` file in the repository (public API, Cython bindings/wrappers, internal scripts, and utilities). The goal is for the level of detail to be proportional to the complexity of the element being documented, not a rigid template copy-pasted everywhere.

### Workflow

**Follow these steps:**

1. If the user doesn't specify scope, ask or infer whether it's: (a) a single file, (b) a whole module, or (c) the entire package.
2. Open each file and classify its elements: module, classes, `__init__`, public methods, private methods (`_foo`), properties, standalone functions, dataclasses, custom exceptions.
3. For each element, choose the **level of detail** (see "Documentation levels" below) based on its complexity — not everything needs full `Args`/`Returns`/`Raises`.
4. Write or fix the docstring following the Google-style format in this skill.
5. Check consistency: same verb mood (imperative or descriptive, see Validation), same type names as the type hints, same section-header casing (`Args:`, `Returns:`, `Raises:`, `Yields:`, `Attributes:`, `Note:`, `Example:`).
6. If the file already has docstrings but they're incomplete or out of sync with the actual function signature, fix them instead of adding a new one alongside.
7. Don't touch code logic. This skill is only for documentation docstrings/comments; any functional change must be confirmed separately.

### Scope: what gets documented and what doesn't

| Element | Documented? |
|---|---|
| Module (`.py` as a unit) | Yes, if the module exposes something public or its purpose isn't obvious from the filename |
| Public classes | Yes, always |
| Public methods/functions | Yes, always |
| `__init__` | Yes, documenting construction `Args:` (don't repeat what the class docstring already says) |
| Dunder methods (`__repr__`, `__eq__`, etc.) | Only if the behavior isn't the trivial expected one |
| Private functions/methods (`_foo`, `__foo`) | Short one-liner if the name isn't self-explanatory; can be omitted if trivial |
| Properties (`@property`) | Yes, one line describing what they return (don't repeat the type if it's already in the type hint) |
| Dataclasses / `NamedTuple` / `TypedDict` | Class docstring + `Attributes:` for each field |
| Custom exceptions | One-liner explaining when it's raised |
| Tests (`test_*.py`) | Only if the test name doesn't explain the scenario covered |
| Internal scripts/utilities (not exposed as API) | Same standard as everything else — don't lower the bar just because they're not public |

---

## Documentation levels

Choose the level based on the actual complexity of the element, not a fixed rule of "every function gets all 3 sections."

### Level 1: One-liner

Use for trivial functions, simple properties, direct wrappers, or non-trivial dunders.

**Format:**
```python
def is_empty(self) -> bool:
    """Return True if the buffer has no pending elements."""
```

**When to use:**
- The signature + name already communicate everything (no ambiguity about units, side effects, or edge cases)
- Simple getters/setters, derived properties
- Single-expression functions with no complex parameters

---

### Level 2: Summary + Args/Returns

Use for functions with non-trivial parameters but no relevant exceptions or generator/async behavior.

**Format:**
```python
def cast_tensor(tensor: Tensor, dtype: DType) -> Tensor:
    """Cast a tensor to the target dtype using the registered cast table.

    Args:
        tensor: Input tensor to cast. Must be contiguous.
        dtype: Target dtype registered in ``ncore::dtypes``.

    Returns:
        A new tensor with the same shape and the requested dtype.
    """
```

**When to use:**
- There are 2+ parameters that need clarification of meaning, units, or constraints
- The return value isn't obvious from the type hint alone (e.g. "new copy" vs. "in-place")

---

### Level 3: Full (Args + Returns + Raises + Note/Example)

Use for public API functions, functions with edge cases, functions that deliberately raise exceptions, or complex entry points (Cython bindings over C/C++, dispatchers, functions with runtime overloads).

**Format:**
```python
def load_checkpoint(path: str, map_location: str | None = None, strict: bool = True) -> dict[str, Tensor]:
    """Load a NovaNN checkpoint from disk into a state dict.

    Reads the on-disk checkpoint format produced by ``save_checkpoint`` and
    reconstructs each tensor via the registered ISA-specific cast table when
    ``map_location`` requires a dtype or device change.

    Args:
        path: Filesystem path to the checkpoint file.
        map_location: Optional device string (e.g. ``"cuda:0"``, ``"cpu"``)
            to remap tensors during load. If ``None``, tensors are restored
            to their original device.
        strict: If True, raise when the checkpoint contains keys not present
            in the current model definition.

    Returns:
        A dict mapping parameter names to their restored ``Tensor`` values.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If ``strict`` is True and the checkpoint has
            mismatched keys.

    Example:
        >>> state = load_checkpoint("model.nova", map_location="cuda:0")
        >>> model.load_state_dict(state)
    """
```

**When to use:**
- Public API function that others will call without reading the source
- Raises specific exceptions the caller must handle
- Has non-obvious side effects (I/O, global state mutation, runtime dispatch)
- It's a binding over C/C++/Rust code and the Python-side behavior isn't obviously 1:1

Use `Yields:` instead of `Returns:` for generators. Use `Attributes:` in the class docstring for dataclasses or classes with relevant public state.

---

## Module docstring

Goes at the top of the file, before imports.

```python
"""Cast dispatch utilities for reduced-precision dtypes.

Exposes the Python-facing wrappers over ``ncore::dtypes`` cast tables,
including BFloat16, Float8_e4m3fn, Float8_e5m2 and Float4_e2m1fn_x2
scalar casting paths.
"""
```

Skip the module docstring if the file is a pure re-export `__init__.py`, or if the filename is already self-explanatory and doesn't expose any API (e.g. `conftest.py` with a single trivial fixture).

## Class docstring

```python
class DAGScheduler:
    """Schedule and execute kernel operations across CPU/GPU thread pools.

    Maintains a dependency graph of pending operations and dispatches
    ready nodes to the appropriate pool (``pool_cpu_`` or ``pool_gpu_``)
    once their inputs are resolved.

    Attributes:
        pool_cpu_: Single-worker pool used for CPU-bound kernels.
        pool_gpu_: Multi-worker pool used for GPU-bound kernels.
    """
```

---

## Validation

- **Style:** Always Google style — `Args:`, `Returns:`, `Yields:`, `Raises:`, `Attributes:`, `Note:`, `Example:` as the only allowed section headers.
- **First line:** summary in imperative or descriptive mood, ending in a period, no longer than ~79 characters. Pick one mode (imperative: "Cast a tensor..."; descriptive: "Casts a tensor...") and keep it consistent across the file/package.
- **Quotes:** always `"""triple double quotes"""`, even for one-liners.
- **Args:** one parameter per line, `name: description.`. Don't repeat the type if it's already in the type hint, unless clarifying a constraint (range, unit, format).
- **Returns/Yields:** describe what the value represents, not just its type.
- **Raises:** only exceptions the function explicitly raises or that propagate in an expected, documented way; don't list unlikely generic exceptions.
- **No repetition:** don't restate the function signature in prose. Don't document something the type hint already makes unambiguous.
- **Don't touch logic:** this skill never modifies behavior, imports, or symbol names — only documentation strings and, if needed, adjacent `#` comments.
- **Binding consistency:** if the function is a Cython wrapper over C/C++/Rust, the docstring describes the contract from the Python side (Python types, Python exceptions), not the underlying C signature.
- **Line length:** max 79 characters per line inside the docstring (PEP 8 / PEP 257).
