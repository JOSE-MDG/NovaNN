"""
Argument processing for Function.apply()

Handles conversion of inputs to raw numpy arrays while tracking which
tensors participate in gradient computation. This module ensures type
consistency and proper gradient flow through operations.
"""

from __future__ import annotations
import numpy as np
import nova
from numpy import dtype
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor


def _is_index_like(obj: Any) -> bool:
    """
    Checks if an object is an indexing construct.

    Indexing constructs (slices, ranges, integer/boolean arrays) should
    not be converted to the base dtype since they're used for addressing,
    not computation.

    Args:
        obj: Object to check.

    Returns:
        True if obj is a slice, range, or array of integers/booleans.

    Examples:
        >>> _is_index_like(slice(0, 5))
        True
        >>> _is_index_like([0, 2, 4])
        True
        >>> _is_index_like([1.0, 2.0])
        False
    """
    if isinstance(obj, (slice, range)):
        return True
    if isinstance(obj, (list, tuple, np.ndarray)):
        if len(obj) == 0:
            return True
        first = obj[0]
        return isinstance(first, (int, np.integer))
    return False


class ArgumentProcessor:
    """
    Processes arguments for Function.apply() calls.

    This class handles the conversion of Function inputs from high-level
    Tensor objects to low-level numpy arrays suitable for forward computation.
    It also tracks which input tensors need gradients for backpropagation.

    Key responsibilities:
    - Extract numpy arrays from Tensors
    - Convert scalars to arrays with consistent dtype
    - Track all Tensors that participate in gradient computation
    - Preserve special dtypes (bool, int) that shouldn't be cast
    - Handle nested structures (lists, tuples, dicts)
    - Skip processing for index-like constructs

    Attributes:
        base_dtype: Target dtype for numerical arrays (typically float32/float64).
        tensors_in_graph: List of Tensors tracked for gradient computation.

    Examples:
        >>> processor = ArgumentProcessor(np.float32)
        >>> args = (tensor1, 5.0, [1, 2, 3])
        >>> raw_args, raw_kwargs = processor.process_args(args, {})
        >>> tensors = processor.get_tracked_tensors()
    """

    def __init__(self, base_dtype: dtype):
        """
        Initializes the argument processor.

        Args:
            base_dtype: Base dtype to cast numerical values to.
        """
        self.base_dtype = base_dtype
        self.tensors_in_graph: list[Tensor] = []

    def process_arg(self, arg: Any) -> Any:
        """
        Processes a single argument for forward pass.

        Conversion rules:
        - Tensor → numpy array (dtype cast if floating-point)
        - ndarray → dtype cast if floating-point
        - float → array with base_dtype
        - int/bool/str/None → unchanged
        - Index-like lists/tuples → unchanged
        - Regular lists/tuples/dicts → recursively processed
        - slice/range → unchanged

        Args:
            arg: Input argument of any type.

        Returns:
            Processed argument ready for forward computation.

        Raises:
            TypeError: If Tensor.data is not a numpy array (indicates bug).
        """
        from nova import Tensor

        if isinstance(arg, Tensor):
            # Track tensor for gradient computation
            self.tensors_in_graph.append(arg)

            # Extract numpy array
            data = arg.data
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    f"Tensor.data must be ndarray, got {type(data)}. "
                    f"This indicates a bug in Tensor implementation."
                )

            # Preserve boolean and integer dtypes
            if data.dtype in (np.bool_, np.int32, np.int64):
                return data

            # Cast to base dtype for numerical stability
            if np.issubdtype(self.base_dtype, np.floating):
                return data.astype(self.base_dtype, copy=False)

            return data
        elif isinstance(arg, np.ndarray):
            # Preserve boolean/integer arrays as-is
            if arg.dtype == np.bool_ or np.issubdtype(arg.dtype, np.integer):
                return arg
            return arg.astype(self.base_dtype, copy=False)

        elif isinstance(arg, float):
            return np.array(arg, dtype=self.base_dtype)

        elif isinstance(arg, (int, bool, str)) or arg is None:
            return arg

        elif isinstance(arg, (list, tuple)):
            # Don't process index-like structures
            if _is_index_like(arg):
                return arg
            # Recursively process containers
            return type(arg)(self.process_arg(a) for a in arg)

        elif isinstance(arg, dict):
            return {k: self.process_arg(v) for k, v in arg.items()}

        elif isinstance(arg, (slice, range)):
            return arg

        # Pass through unknown types unchanged
        return arg

    def process_args(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """
        Processes all positional and keyword arguments.

        Args:
            args: Tuple of positional arguments.
            kwargs: Dictionary of keyword arguments.

        Returns:
            Tuple of (processed_args, processed_kwargs).
        """
        raw_args = tuple(self.process_arg(a) for a in args)
        raw_kwargs = {k: self.process_arg(v) for k, v in kwargs.items()}
        return raw_args, raw_kwargs

    def get_tracked_tensors(self) -> list[Tensor]:
        """
        Returns list of tensors tracked during argument processing.

        These tensors will be stored as inputs to the operation for
        gradient computation during the backward pass.

        Returns:
            List of Tensors that participated in the operation.
        """
        return self.tensors_in_graph


def determine_base_dtype(args: tuple) -> dtype:
    """
    Determines the base dtype for an operation from its arguments.

    Scans arguments prioritizing floating-point types over integers to ensure
    proper type promotion in mixed operations (e.g., int * float → float).

    Priority order:
    1. Complex floating-point (complex64, complex128)
    2. Floating-point (float16, float32, float64)
    3. Integer (int32, int64) - only if no floats found
    4. Default to float32 if no Tensor found

    Args:
        args: Tuple of arguments to scan.

    Returns:
        Base dtype to use for the operation.

    Examples:
        >>> x = nova.tensor([1.0], dtype=nova.float32)
        >>> y = nova.tensor([2], dtype=nova.int64)
        >>> dtype = determine_base_dtype((y, x))  # Order doesn't matter
        >>> print(dtype)  # float32 (prioritizes float over int)
    """
    from nova import Tensor

    # First pass: look for complex types (highest priority)
    for arg in args:
        if isinstance(arg, Tensor) and np.issubdtype(arg.dtype, np.complexfloating):
            return arg.dtype

    # Second pass: look for floating-point types
    for arg in args:
        if isinstance(arg, Tensor) and np.issubdtype(arg.dtype, np.floating):
            return arg.dtype

    # Third pass: look for integer types (only if no floats found)
    for arg in args:
        if isinstance(arg, Tensor) and np.issubdtype(arg.dtype, np.integer):
            return arg.dtype

    # Default to float32 if no Tensor found
    return nova.float32
