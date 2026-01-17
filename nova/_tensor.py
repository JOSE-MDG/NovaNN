from __future__ import annotations
import nova
import numpy as np
import traceback
from numpy import ndarray
from typing import TYPE_CHECKING, Any, Optional, Self
from nova._interfaces._base_tensor import TensorBase
from nova.utils import registry_class, ensure_tensor
from nova.utils.logger import logger
from nova.autograd.engine import _backward
from nova.utils.hooks import HooksHandle

if TYPE_CHECKING:
    from nova.autograd.function import Function
    from nova._typing import Dtype, Hook, Dim
    from nova.autograd.engine import Context


@registry_class
class Tensor(TensorBase):
    """
    Main tensor class with automatic differentiation support.

    Tensor is the core data structure in Nova, similar to PyTorch's torch.Tensor.
    It wraps a numpy array and provides:
    - Automatic differentiation via the autograd system
    - Rich set of operations (arithmetic, linear algebra, etc.)
    - GPU-like API (though currently CPU-only)
    - Gradient tracking and backpropagation

    Attributes:
        data: Underlying numpy array containing the tensor's values.
        dtype: Data type of tensor elements.
        requires_grad: If True, operations are tracked for gradient computation.
        grad: Accumulated gradients (None until backward is called).
        grad_fn: Function that created this tensor (None for leaf tensors).
        shape: Dimensions of the tensor.
        device: Always 'cpu' (GPU support planned).

    Examples:
        >>> # Create tensors
        >>> x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> w = nova.tensor([0.5, -0.5, 1.0], requires_grad=True)

        >>> # Forward pass
        >>> y = (x * w).sum()

        >>> # Backward pass
        >>> y.backward()
        >>> print(x.grad)  # [0.5, -0.5, 1.0]
        >>> print(w.grad)  # [1.0, 2.0, 3.0]
    """

    __slots__ = [
        "_data_internal",
        "_dtype_internal",
        "requires_grad",
        "grad_fn",
        "grad",
        "copy",
        "_is_leaf",
        "_backward_hooks",
        "_retain_grad",
        "_inputs",
        "_ctx",
    ]

    _data_internal: ndarray
    _dtype_internal: Dtype
    requires_grad: bool
    grad_fn: Optional[Function]
    grad: Optional[ndarray]
    copy: bool
    _is_leaf: bool
    _backward_hooks: list[Hook]
    _retain_grad: bool
    _inputs: list[Tensor]
    _ctx: Context

    def __init__(
        self,
        data: Any,
        dtype: Optional[Dtype] = None,
        requires_grad: bool = False,
        grad_fn: Optional[Function] = None,
        copy: bool = True,
    ):
        """
        Constructs a new Tensor.

        Args:
            data: Data to initialize the tensor. Can be a list, numpy array,
                scalar, or another Tensor.
            dtype: Desired data type. If None, defaults to float32.
            requires_grad: If True, tracks operations for gradient computation.
            grad_fn: Function that created this tensor (internal use).
            copy: If True, creates a copy of the data. If False, may share
                memory with the input (use with caution).

        Examples:
            >>> x = nova.Tensor([1, 2, 3])  # From list
            >>> y = nova.Tensor(np.array([1, 2, 3]))  # From numpy
            >>> z = nova.Tensor(5.0, requires_grad=True)  # Scalar with grad
        """
        self._dtype_internal: Dtype = dtype if dtype is not None else nova.float32

        # Handle Tensor input
        if isinstance(data, Tensor):
            if copy:
                data = data.data.astype(self._dtype_internal, copy=True)
            else:
                if data.data.dtype != self._dtype_internal:
                    data = data.data.astype(self._dtype_internal)
                else:
                    data = data.data

        # Handle numpy array input
        elif isinstance(data, ndarray):
            if copy:
                data = data.astype(self._dtype_internal, copy=True)
            else:
                if data.dtype != self._dtype_internal:
                    data = data.astype(self._dtype_internal)
                else:
                    data = data

        # Handle other inputs (lists, scalars, etc.)
        else:
            data = np.array(data, dtype=self._dtype_internal)

        super().__init__()
        self.data: ndarray = data
        self.requires_grad: bool = requires_grad
        self.grad_fn: Optional[Function] = grad_fn
        self._inputs: list[Tensor] = []
        self._ctx: Optional[Context] = None
        self.grad: Optional[ndarray] = None
        self._is_leaf: bool = True if grad_fn is None else False
        self._retain_grad: bool = False
        self._backward_hooks: list[Hook] = []

    # Comparison operators

    def __eq__(self, other) -> Tensor:
        """Element-wise equality comparison."""
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data == other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __req__(self, other) -> Tensor:
        """Reverse equality comparison."""
        return self.__eq__(other)

    def __ne__(self, other) -> Tensor:
        """Element-wise inequality comparison."""
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data != other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __rne__(self, other) -> Tensor:
        """Reverse inequality comparison."""
        return self.__ne__(other)

    def __lt__(self, other) -> Tensor:
        """Element-wise less than comparison."""
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data < other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __rlt__(self, other) -> Tensor:
        """Reverse less than comparison."""
        return self.__gt__(other)

    def __le__(self, other) -> Tensor:
        """Element-wise less than or equal comparison."""
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data <= other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __rle__(self, other) -> Tensor:
        """Reverse less than or equal comparison."""
        return self.__ge__(other)

    def __gt__(self, other) -> Tensor:
        """Element-wise greater than comparison."""
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data > other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __rgt__(self, other) -> Tensor:
        """Reverse greater than comparison."""
        return self.__lt__(other)

    def __ge__(self, other) -> Tensor:
        """Element-wise greater than or equal comparison."""
        other_tensor = ensure_tensor(other)
        return Tensor(
            self.data >= other_tensor.data, dtype=nova.bool, requires_grad=False
        )

    def __rge__(self, other) -> Tensor:
        """Reverse greater than or equal comparison."""
        return self.__le__(other)

    def eq(self, other) -> Tensor:
        """Element-wise equality (alias for ==)."""
        return self.__eq__(other)

    def ne(self, other) -> Tensor:
        """Element-wise inequality (alias for !=)."""
        return self.__ne__(other)

    def lt(self, other) -> Tensor:
        """Element-wise less than (alias for <)."""
        return self.__lt__(other)

    def le(self, other) -> Tensor:
        """Element-wise less than or equal (alias for <=)."""
        return self.__le__(other)

    def gt(self, other) -> Tensor:
        """Element-wise greater than (alias for >)."""
        return self.__gt__(other)

    def ge(self, other) -> Tensor:
        """Element-wise greater than or equal (alias for >=)."""
        return self.__ge__(other)

    # Special methods

    def __invert__(self) -> Tensor:
        """Bitwise NOT operation."""
        return Tensor(~self.data, dtype=nova.bool, requires_grad=False)

    def __getstate__(self):
        """Return state for pickling.

        Needed because the class uses __slots__.
        """
        state = {}
        for slot in self.__slots__:
            if hasattr(self, slot):
                state[slot] = getattr(self, slot)
        return state

    def __setstate__(self, state):
        """Restore state from pickling.

        Needed because the class uses __slots__.
        """
        for slot, value in state.items():
            setattr(self, slot, value)

    def __hash__(self):
        """Hash based on object identity."""
        return id(self)

    def __len__(self) -> int:
        """Returns length of first dimension."""
        return len(self.data)

    def __bool__(self) -> bool:
        """
        Converts single-element tensor to boolean.

        Raises:
            RuntimeError: If tensor has more than one element.
        """
        if self.numel() != 1:
            raise RuntimeError(
                "The truth value of a tensor with more than one element is ambiguous. "
                "Use nova.any() or nova.all() for reduction operations."
            )
        return bool(self.item())

    # Argument operations

    def argmax(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        """
        Returns indices of maximum values along a dimension.

        Args:
            dim: Dimension to reduce. If None, returns index in flattened array.
            keepdims: If True, retains reduced dimension with size 1.

        Examples:
            >>> x = nova.tensor([[1, 3, 2], [4, 2, 5]])
            >>> x.argmax(dim=1)
            tensor([1, 2])
        """
        return Tensor(
            self.data.argmax(axis=dim, keepdims=keepdims),
            dtype=nova.long,
            requires_grad=False,
        )

    def argmin(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        """
        Returns indices of minimum values along a dimension.

        Args:
            dim: Dimension to reduce. If None, returns index in flattened array.
            keepdims: If True, retains reduced dimension with size 1.
        """
        return Tensor(
            self.data.argmin(axis=dim, keepdims=keepdims),
            dtype=nova.long,
            requires_grad=False,
        )

    def argsort(self, dim: Optional[Dim] = None, kind=None, order=None) -> Tensor:
        """
        Returns indices that would sort the tensor.

        Args:
            dim: Dimension to sort along. If None, sorts flattened array.
            kind: Sorting algorithm ('quicksort', 'mergesort', 'heapsort', 'stable').
            order: For structured arrays, which fields to compare first.
        """
        return Tensor(
            self.data.argsort(axis=dim, kind=kind, order=order),
            dtype=nova.long,
            requires_grad=False,
        )

    def argwhere(self) -> Tensor:
        """Returns indices of non-zero elements."""
        return Tensor(np.argwhere(self.data), dtype=nova.long, requires_grad=False)

    # Statistical operations

    def std(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        """
        Computes standard deviation along a dimension.

        Args:
            dim: Dimension to reduce. If None, computes over all elements.
            keepdims: If True, retains reduced dimension with size 1.
        """
        from nova.autograd._ops import std

        return std(self, dim=dim, keepdims=keepdims)

    # Utility operations

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Self:
        """
        Flattens continuous dimensions within a range.

        Args:
            start_dim: First dimension to flatten (default: 0)
            end_dim: Last dimension to flatten (default: -1, last dim)

        Returns:
            Tensor with flattened dimensions
        Examples:
            >>> x = nova.randn(2, 3, 4, 5)
            >>> x.flatten(1, 2).shape
            (2, 12, 5)  # flattened dims 1 and 2: 3*4=12

            >>> x.flatten(0, -1).shape
            (120,)  # all dimensions flattened: 2*3*4*5=120

            >>> x.flatten(1).shape
            (2, 60)  # from dim 1 to the end: 3*4*5=60
        """
        # Normalize negative indices
        ndim = self.ndim
        start_dim = start_dim if start_dim >= 0 else ndim + start_dim
        end_dim = end_dim if end_dim >= 0 else ndim + end_dim

        # Validations
        if start_dim < 0 or start_dim >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], "
                f"but got {start_dim})"
            )
        if end_dim < 0 or end_dim >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], "
                f"but got {end_dim})"
            )
        if start_dim > end_dim:
            raise RuntimeError(
                f"flatten() has invalid args: start_dim cannot come after end_dim "
                f"(got start_dim={start_dim}, end_dim={end_dim})"
            )

        # Special case: if start_dim == end_dim, do nothing
        if start_dim == end_dim:
            return self

        # Calculate new shape
        shape = list(self.shape)

        # Dims before the range to flatten
        new_shape = shape[:start_dim]

        # Calculate product of dimensions to flatten
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= shape[i]
        new_shape.append(flattened_size)

        # Dims after the range to flatten
        new_shape.extend(shape[end_dim + 1 :])

        # Reshape
        return self.reshape(*tuple(new_shape))

    def detach(self) -> Tensor:
        """
        Returns a new tensor detached from the computation graph.

        The returned tensor will never require gradients and won't
        be tracked by autograd.

        Examples:
            >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> y = x * 2
            >>> z = y.detach()  # Breaks gradient flow
            >>> z.requires_grad
            False
        """
        return Tensor(self.data, dtype=self.dtype, requires_grad=False)

    def item(self) -> float | int:
        """
        Extracts the scalar value from a single-element tensor.

        Returns:
            Python scalar (int or float).

        Raises:
            ValueError: If tensor has more than one element.

        Examples:
            >>> x = nova.tensor([5.0])
            >>> x.item()
            5.0
        """
        if self.numel() > 1:
            raise ValueError(
                "only one element tensors can be converted to Python scalars."
            )

        return self.data.item()

    def all(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        """Tests if all elements are True along a dimension."""
        return Tensor(
            self.data.all(dim, keepdims=keepdims), dtype=nova.bool, requires_grad=False
        )

    def any(self, dim: Optional[Dim] = None, keepdims: bool = False) -> Tensor:
        """Tests if any element is True along a dimension."""
        return Tensor(
            self.data.any(dim, keepdims=keepdims), dtype=nova.bool, requires_grad=False
        )

    def clone(self) -> Tensor:
        """
        Returns a copy of the tensor with the same data.

        If the original tensor requires gradients, the cloned tensor
        will be connected to the computation graph and gradients will
        flow back through the clone operation.

        Examples:
            >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> y = x.clone()
            >>> y[0] = 999  # Doesn't affect x
            >>> loss = y.sum()
            >>> loss.backward()
            >>> print(x.grad)  # Gradients flow through
            [1., 1.]
        """
        from nova.autograd._ops import Clone

        if self.requires_grad:
            return Clone.apply(self)
        else:
            return Tensor(self.data.copy(), dtype=self.dtype, requires_grad=False)

    # Type conversion and casting

    def type(self, dtype: Optional[Dtype] = None) -> Tensor | str:
        """
        Returns the type if dtype is None, or casts to dtype.

        Examples:
            >>> x.type()
            'nova.float32'
            >>> x.type(nova.long)
            tensor([...], dtype=int64)
        """
        if dtype is None:
            return f"nova.{np.dtype(self.dtype).name}"
        return self.to(dtype)

    def is_contiguous(self) -> bool:
        """Returns True if tensor is contiguous in memory (C-order)."""
        return self.data.flags["C_CONTIGUOUS"]

    def contiguous(self) -> Tensor:
        """Returns a contiguous copy if needed, otherwise returns self."""
        if self.is_contiguous():
            return self
        return Tensor(
            np.ascontiguousarray(self.data),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )

    def to(self, dtype: Dtype) -> Tensor:
        """
        Casts tensor to a different dtype.

        Args:
            dtype: Target data type.

        Returns:
            New tensor with specified dtype.

        Notes:
            If tensor requires_grad, computational graph linkage is preserved.
        """
        if dtype == self.dtype:
            return self
        data = self.data.astype(dtype, copy=True)

        out = Tensor(data, dtype=dtype, requires_grad=self.requires_grad)

        # Preserve computational graph for gradient flow
        if self.requires_grad:
            out._inputs = self._inputs
            out.grad_fn = self.grad_fn
            out._ctx = self._ctx
            out._is_leaf = False

        return out

    def float(self) -> Tensor:
        """Casts to float32."""
        return self.to(nova.float32)

    def double(self) -> Tensor:
        """Casts to float64."""
        return self.to(nova.double)

    def int(self) -> Tensor:
        """Casts to int32."""
        return self.to(nova.int)

    def long(self) -> Tensor:
        """Casts to int64."""
        return self.to(nova.long)

    def numpy(self) -> ndarray:
        """Returns a copy of the tensor as a numpy array."""
        return self.data.copy()

    def allclose(self, other, rtol=1e-5, atol=1e-8, equal_nan: bool = False) -> bool:
        """
        Checks if tensors are element-wise equal within tolerance.

        Args:
            other: Tensor to compare against.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            equal_nan: If True, NaNs are considered equal.

        Returns:
            True if all elements are close.
        """
        return np.allclose(
            self.data,
            ensure_tensor(other).data,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    # In-place initialization methods

    def normal_(self, mean: float = 0, std: float = 1) -> None:
        """
        Fills tensor with values from normal distribution (in-place).

        Args:
            mean: Mean of the distribution.
            std: Standard deviation.

        Raises:
            RuntimeError: If tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data = np.random.normal(loc=mean, scale=std, size=self.data.shape).astype(
            self.dtype
        )

    def uniform_(self, low: float = -1, high: float = 1) -> None:
        """
        Fills tensor with values from uniform distribution (in-place).

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).

        Raises:
            RuntimeError: If tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data = np.random.uniform(low=low, high=high, size=self.data.shape).astype(
            self.dtype
        )

    def random_(self) -> None:
        """
        Fills tensor with random values in [0, 1) (in-place).

        Raises:
            RuntimeError: If tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data = np.random.rand(*self.data.shape).astype(self.dtype)

    def zero_(self) -> None:
        """
        Fills tensor with zeros (in-place).

        Raises:
            RuntimeError: If tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data.fill(0.0)

    def ones_(self) -> None:
        """
        Fills tensor with ones (in-place).

        Raises:
            RuntimeError: If tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data.fill(1.0)

    def fill_(self, value: Any) -> None:
        """
        Fills tensor with a scalar value (in-place).

        Args:
            value: Scalar to fill with.

        Raises:
            RuntimeError: If tensor requires gradients.
        """
        if self.requires_grad:
            raise RuntimeError(
                "Cannot perform inplace operation on a tensor that requires gradients"
            )

        self.data.fill(value)

    def copy_(self, src: Tensor) -> Self:
        """
        Copies data from another tensor (in-place).

        Args:
            src: Source tensor to copy from.

        Returns:
            self for method chaining.

        Raises:
            ValueError: If shapes don't match.

        Examples:
            >>> x = nova.zeros(3)
            >>> y = nova.ones(3)
            >>> x.copy_(y)
            >>> print(x)  # tensor([1., 1., 1.])
        """
        src = ensure_tensor(src)

        src_data = src.data

        if src.data.shape != self.data.shape:
            raise ValueError(f"Shape mismatch: {self.data.shape} vs {src_data.shape}")

        if src_data.dtype != self.data.dtype:
            src_data = src_data.astype(self.data.dtype)

        np.copyto(dst=self.data, src=src_data)

        return self

    # Gradient management

    def requires_grad_(self, mode: bool) -> None:
        """
        Sets requires_grad flag (in-place).

        Args:
            mode: If True, enables gradient tracking.
        """
        self.requires_grad = mode

    def retain_grad(self) -> Self:
        """
        Enables gradient retention for non-leaf tensor.

        By default, only leaf tensors retain gradients after backward.
        Call this method on intermediate tensors to keep their gradients.

        Returns:
            self for method chaining.

        Raises:
            RuntimeError: If tensor doesn't require gradients.

        Examples:
            >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> y = x * 2
            >>> y.retain_grad()  # Keep y.grad after backward
            >>> z = y.sum()
            >>> z.backward()
            >>> print(y.grad)  # [1., 1.]
        """
        if not self.requires_grad:
            raise RuntimeError("Only tensors with requires_grad can retain gradients")

        self._retain_grad = True

        return self

    def register_hook(self, hook: Hook) -> HooksHandle:
        """
        Registers a backward hook on this tensor.

        Hooks are called during backward pass and can inspect or modify gradients.

        Args:
            hook: Callable that receives gradient and optionally returns modified gradient.

        Returns:
            Handle that can be used to remove the hook.

        Raises:
            RuntimeError: If tensor doesn't require gradients.

        Examples:
            >>> def print_grad(grad):
            ...     print(f"Gradient: {grad}")
            ...     return grad
            >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> handle = x.register_hook(print_grad)
            >>> y = (x ** 2).sum()
            >>> y.backward()  # Prints: Gradient: [2., 4.]
        """
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot register a hook on a tensor that doesn't require gradients"
            )

        self._backward_hooks.append(hook)
        return HooksHandle(self._backward_hooks, hook)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Clears the gradient.

        Args:
            set_to_none: If True, sets grad to None (saves memory).
                If False, fills grad with zeros.

        Examples:
            >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> y = x.sum()
            >>> y.backward()
            >>> x.zero_grad()
            >>> print(x.grad)  # None
        """
        if set_to_none:
            self.grad = None
        else:
            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=self.data.dtype)
            else:
                self.grad.fill(0.0)

    def _apply_hooks(self, grad: ndarray) -> ndarray:
        """
        Applies registered backward hooks to gradient.

        Args:
            grad: Incoming gradient.

        Returns:
            Potentially modified gradient after all hooks.

        Notes:
            Hooks are applied sequentially in registration order.
            If a hook returns None, gradient passes through unchanged.
        """
        try:
            current_grad = grad

            for hook in self._backward_hooks:
                result = hook(current_grad)

                if result is not None:
                    if result.shape != grad.shape:
                        raise ValueError(
                            f"Hook returned gradient with shape {result.shape}, "
                            f"expected {current_grad.shape}"
                        )

                    current_grad = result

            return current_grad
        except Exception as e:
            lines = [line for line in traceback.format_exception(e)]
            logger.error("Error executing backward hook\n\n")
            print(*lines)

    def backward(
        self, gradient: Optional[ndarray | Tensor] = None, retain_graph: bool = False
    ) -> None:
        """
        Computes gradients via backpropagation.

        This method initiates the backward pass through the computational graph,
        computing gradients of this tensor with respect to all leaf tensors that
        have requires_grad=True.

        Args:
            gradient: Initial gradient (dL/d_output). If None, assumes scalar
                output and uses gradient of ones. For non-scalar outputs, must
                be provided explicitly.
            retain_graph: If True, keeps computational graph for multiple backward
                passes. If False (default), frees graph memory after backward.

        Raises:
            RuntimeError: If tensor doesn't require gradients or if gradient is
                None for non-scalar output.

        Examples:
            >>> # Scalar output (gradient inferred)
            >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> y = (x ** 2).sum()
            >>> y.backward()
            >>> print(x.grad)  # [2., 4.]

            >>> # Non-scalar output (gradient required)
            >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> y = x ** 2
            >>> y.backward(nova.tensor([1.0, 1.0]))
            >>> print(x.grad)  # [2., 4.]

            >>> # Multiple backward passes
            >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> y = (x ** 2).sum()
            >>> y.backward(retain_graph=True)
            >>> x.zero_grad()
            >>> y.backward()  # Graph still exists
        """
        if not self.requires_grad:
            raise RuntimeError(
                "An attempt was made to calculate gradients for a tensor that does not require gradients."
            )

        # Handle implicit gradient for scalar outputs
        if gradient is None:
            if self.numel() > 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs"
                )
            gradient = np.ones_like(self.data, dtype=nova.float32)
        else:
            # Convert Tensor gradient to numpy array
            gradient = (
                gradient.data.astype(nova.float32)
                if isinstance(gradient, Tensor)
                else np.asarray(gradient, dtype=nova.float32)
            )

        try:
            _backward(self, gradient=gradient, retain_graph=retain_graph)
        except Exception as e:
            lines = [line for line in traceback.format_exception(e)]
            logger.error("Error during the graph creation\n\n")
            print(*lines)

    # String representations

    def __repr__(self) -> str:
        """
        Returns detailed string representation showing all tensor metadata.

        Examples:
            >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> repr(x)
            'tensor([1., 2.], requires_grad=True, grad_fn=None, dtype=float32)'
        """
        prefix = "tensor("
        tensor_str = np.array2string(self.data, separator=", ", prefix=prefix)

        result = f"{prefix}{tensor_str}"
        result += f", requires_grad={self.requires_grad}"
        result += f", grad_fn={repr(self.grad_fn)}"

        dtype_str = np.dtype(self.dtype).name
        result += f", dtype={dtype_str}"
        result += ")"
        return result

    def __str__(self) -> str:
        """
        Returns concise string representation for printing.

        Examples:
            >>> x = nova.tensor([1.0, 2.0])
            >>> print(x)
            tensor([1., 2.])

            >>> y = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> print(y)
            tensor([1., 2.], requires_grad=True)
        """
        prefix = "tensor("

        array_str = np.array2string(self.data, separator=", ", prefix=prefix)

        result = f"{prefix}{array_str}"
        if self.requires_grad:
            result += f", requires_grad={self.requires_grad}"
        result += ")"
        return result
