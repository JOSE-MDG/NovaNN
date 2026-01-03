"""
Argument processing for Function.apply()
Handles conversion of inputs to raw numpy arrays while tracking tensors.
"""

from __future__ import annotations
import numpy as np
import nova
from numpy import dtype
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor


def _is_index_like(obj: Any) -> bool:
    """Check if object is an indexing construct (slice, range, index array)."""
    if isinstance(obj, (slice, range)):
        return True
    if isinstance(obj, (list, tuple, np.ndarray)):
        if len(obj) == 0:
            return True
        first = obj[0]
        return isinstance(first, (int, bool, np.integer, np.bool_))
    return False


class ArgumentProcessor:
    """
    Processes arguments for Function.apply()
    Converts Tensors to ndarrays and tracks which inputs need gradients.
    """

    def __init__(self, base_dtype: dtype):
        self.base_dtype = base_dtype
        self.tensors_in_graph: list[Tensor] = []

    def process_arg(self, arg: Any) -> Any:
        """
        Process a single argument for forward pass.

        Args:
            arg: Input argument (Tensor, ndarray, scalar, etc.)

        Returns:
            Processed argument (usually ndarray or scalar)
        """
        from nova import Tensor

        if isinstance(arg, Tensor):
            # Track this tensor for gradient computation
            self.tensors_in_graph.append(arg)

            # Extract numpy array
            data = arg.data
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    f"Tensor.data must be ndarray, got {type(data)}. "
                    f"This indicates a bug in Tensor implementation."
                )

            # Keep boolean and integer dtypes as-is
            if data.dtype in (np.bool_, np.int32, np.int64):
                return data

            # Convert to base dtype for numerical stability
            return data.astype(self.base_dtype, copy=False)

        elif isinstance(arg, np.ndarray):
            # Keep boolean/integer arrays as-is
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

        # Pass through unknown types
        return arg

    def process_args(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """
        Process all args and kwargs.

        Returns:
            Tuple of (processed_args, processed_kwargs)
        """
        raw_args = tuple(self.process_arg(a) for a in args)
        raw_kwargs = {k: self.process_arg(v) for k, v in kwargs.items()}
        return raw_args, raw_kwargs

    def get_tracked_tensors(self) -> list[Tensor]:
        """Get list of tensors that were tracked during processing."""
        return self.tensors_in_graph


def determine_base_dtype(args: tuple) -> dtype:
    """
    Determine the base dtype for the operation from arguments.

    Args:
        args: Arguments to scan

    Returns:
        Base dtype to use (defaults to float32 if no Tensor found)
    """
    from nova import Tensor

    base_dtype = None
    for arg in args:
        if isinstance(arg, Tensor):
            base_dtype = arg.dtype
            break
    if base_dtype is None:
        base_dtype = nova.float32

    return base_dtype
