"""
Data type definitions for NovaNN.

This module defines the canonical numeric, boolean, and complex data types used
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
- Complex types are supported for advanced numerical computing.
- Unsigned integer types are available for specific use cases.

Type categories:
- **Unsigned integers**: uint8, uint16, uint32, uint64
- **Signed integers**: int8, int16 (short), int32 (int), int64 (long)
- **Floating point**: float16 (half), float32, float64 (double), float128
- **Complex**: complex64 (cfloat), complex128 (cdouble), complex256 (clongdouble)
- **Boolean**: bool
- **Other**: byte (alias for int8)

Examples:
    >>> import nova
    >>> x = nova.tensor([1, 2, 3], dtype=nova.int)
    >>> x.type()
    'nova.int32'

    >>> y = nova.tensor([True, False], dtype=nova.bool)
    >>> y.type()
    'nova.bool'

    >>> z = nova.tensor([1.0 + 2.0j, 3.0 + 4.0j], dtype=nova.complex64)
    >>> z.type()
    'nova.complex64'
"""

import sys
from numpy import (
    # Unsigned integers
    uint8 as _uint8,
    uint16 as _uint16,
    uint32 as _uint32,
    uint64 as _uint64,
    # Signed integers
    int8 as _int8,
    int16 as _int16,
    int32 as _int32,
    int64 as _int64,
    # Floating point
    float16 as _float16,
    float32 as _float32,
    float64 as _float64,
    float128 as _float128,
    # Complex
    complex64 as _complex64,
    complex128 as _complex128,
    complex256 as _complex256,
    # Boolean
    bool_ as _bool,
)

# Unsigned Integer Types

uint8 = _uint8
"""Unsigned 8-bit integer (0 to 255)."""

uint16 = _uint16
"""Unsigned 16-bit integer (0 to 65,535)."""

uint32 = _uint32
"""Unsigned 32-bit integer (0 to 4,294,967,295)."""

uint64 = _uint64
"""Unsigned 64-bit integer (0 to 18,446,744,073,709,551,615)."""

# Signed Integer Types

int8 = _int8
"""Signed 8-bit integer (-128 to 127)."""

byte = _int8
"""Alias for int8. Signed 8-bit integer (-128 to 127)."""

int16 = _int16
"""Signed 16-bit integer (-32,768 to 32,767)."""

short = _int16
"""Alias for int16. Signed 16-bit integer (-32,768 to 32,767)."""

int32 = _int32
"""Signed 32-bit integer (-2,147,483,648 to 2,147,483,647)."""

int = _int32
"""Alias for int32. Signed 32-bit integer (default integer type)."""

int64 = _int64
"""Signed 64-bit integer (-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807)."""

long = _int64
"""Alias for int64. Signed 64-bit integer (default for large integers)."""

# Floating Point Types

float16 = _float16
"""Half precision (16-bit) floating point number."""

half = _float16
"""Alias for float16. Half precision floating point."""

float32 = _float32
"""Single precision (32-bit) floating point number (default float type)."""

float = _float32
"""Alias for float32. Single precision floating point."""

float64 = _float64
"""Double precision (64-bit) floating point number."""

double = _float64
"""Alias for float64. Double precision floating point."""

# Select appropriate dtypes based on OS support
if not sys.platform.startswith("win32"):
    float128 = _float128
    """Quadruple precision (128-bit) floating point number (extended precision)."""

    longdouble = _float128
    """Alias for float128. Extended precision floating point."""

    complex256 = _complex256
    """Complex number with two 128-bit floats (real and imaginary parts)."""

    clongdouble = _complex256
    """Alias for complex256. Complex number with float128 components."""
else:
    # Fallback for Windows where 128-bit types are not supported
    float128 = _float64
    """Quadruple precision (128-bit) floating point number (extended precision). 
    Note: On Windows, this falls back to 64-bit float."""

    longdouble = _float64
    """Alias for float128. Extended precision floating point.
    Note: On Windows, this falls back to 64-bit float."""

    complex256 = _complex128
    """Complex number with two 128-bit floats (real and imaginary parts).
    Note: On Windows, this falls back to complex128."""

    clongdouble = _complex128
    """Alias for complex256. Complex number with float128 components.
    Note: On Windows, this falls back to complex128."""

# Complex Types

complex64 = _complex64
"""Complex number with two 32-bit floats (real and imaginary parts)."""

cfloat = _complex64
"""Alias for complex64. Complex number with float32 components."""

complex128 = _complex128
"""Complex number with two 64-bit floats (real and imaginary parts)."""

cdouble = _complex128
"""Alias for complex128. Complex number with float64 components."""

# Boolean Type

bool = _bool
"""Boolean type (True or False)."""

__all__ = [
    # Unsigned integers
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    # Signed integers
    "int8",
    "byte",
    "int16",
    "short",
    "int32",
    "int",
    "int64",
    "long",
    # Floating point
    "float16",
    "half",
    "float32",
    "float",
    "float64",
    "double",
    "float128",
    "longdouble",
    # Complex
    "complex64",
    "cfloat",
    "complex128",
    "cdouble",
    "complex256",
    "clongdouble",
    # Boolean
    "bool",
]
