"""
Data type definitions for NovaNN.

This module defines the canonical numeric and boolean data types used
throughout the NovaNN framework. All types are thin aliases of NumPy
scalar dtypes and are provided to offer a consistent, framework-level
API similar to libraries such as PyTorch.

The exposed names intentionally follow common deep learning conventions
(e.g., `int`, `long`, `half`) rather than Python built-ins.

Important notes:
- These names are aliases, not new types.
- All aliases map directly to NumPy dtypes.
- Some names (e.g., `int`, `bool`) shadow Python built-ins by design.
  This is intentional and scoped to the NovaNN API.

Examples:
    >>> import nova
    >>> x = nova.tensor([1, 2, 3], dtype=nova.int)
    >>> x.type()
    'nova.int32'

    >>> y = nova.tensor([True, False], dtype=nova.bool)
    >>> y.type()
    'nova.bool'
"""

from numpy import (
    uint8 as _uint8,
    int8 as _int8,
    int16 as _int16,
    int32 as _int32,
    int64 as _int64,
    float16 as _float16,
    float32 as _float32,
    double as _double,
    float128 as _float128,
    bool_ as _bool,
)

# Integer types
uint8 = _uint8
int8 = _int8
short = _int16
int = _int32
long = _int64

# Floating point types
half = _float16
float32 = _float32
double = _double
float128 = _float128

# Boolean type
bool = _bool

__all__ = [
    "uint8",
    "int8",
    "short",
    "int",
    "long",
    "half",
    "float32",
    "double",
    "float128",
    "bool",
]
