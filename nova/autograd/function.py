from __future__ import annotations
import nova
import numpy as np
from .engine import Context
from .utils import ArgumentProcessor, determine_base_dtype
from typing import Any, TYPE_CHECKING, Type, TypeVar
from abc import ABC, ABCMeta

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Gradients

TFunction = TypeVar("TFunction", bound="Function")


class FunctionMeta(ABCMeta):
    """
    Metaclass for Function that customizes its string representation.

    Provides a clean repr format for Function classes when displayed,
    showing them as "<FunctionNameBackward>" to indicate their role
    in the backward pass.
    """

    def __repr__(cls) -> str:
        return f"<{cls.__name__}Backward>"


class Function(ABC, metaclass=FunctionMeta):
    """
    Base class for all differentiable operations in Nova's autograd system.

    Function defines the interface for operations that support automatic
    differentiation. Each operation must implement:
    - forward(): Computes the operation's output
    - backward(): Computes gradients with respect to inputs

    The apply() method orchestrates the forward pass, tracks tensors for
    gradient computation, and constructs the computational graph.

    Key responsibilities:
    - Execute forward computation on raw numpy arrays
    - Track inputs that require gradients
    - Save intermediate values needed for backward pass via Context
    - Build computational graph nodes for backpropagation
    - Handle dtype casting and numerical stability

    Examples:
        >>> class Square(Function):
        ...     @staticmethod
        ...     def forward(ctx, x):
        ...         ctx.save_for_backward(x)
        ...         return x ** 2
        ...
        ...     @staticmethod
        ...     def backward(ctx, grad_output):
        ...         x, = ctx.saved_tensors
        ...         return 2 * x * grad_output

        >>> x = nova.tensor([2.0, 3.0], requires_grad=True)
        >>> y = Square.apply(x)
        >>> y.backward()
        >>> print(x.grad)  # [4.0, 6.0]
    """

    @staticmethod
    def forward(ctx: Context, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Computes the forward pass of the operation.

        This method receives raw numpy arrays (not Tensors) and performs
        the actual computation. It can save intermediate values to ctx
        for use in the backward pass.

        Args:
            ctx (Context): Context object for saving tensors/values needed in backward.
            *args (Any): Input arrays and parameters for the operation.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Result of the operation as a numpy array.

        Notes:
            - Must be implemented by subclasses
            - Receives numpy arrays, not Tensors
            - Use ctx.save_for_backward() to store values for backward pass
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Gradients:
        """
        Computes gradients with respect to inputs (backward pass).

        This method receives the gradient flowing from downstream operations
        and computes gradients with respect to each input that requires_grad.

        Args:
            ctx (Context): Context containing saved tensors from forward pass.
            grad_output (ndarray): Gradient of the loss with respect to this operation's output.

        Returns:
            Tuple of gradients, one for each input. Return None for inputs
            that don't need gradients (e.g., integer indices, constants).

        Notes:
            - Must be implemented by subclasses
            - Retrieved saved values via ctx.saved_tensors
            - Must handle broadcasting correctly for gradients
            - Return tuple even for single input operations

        Examples:
            >>> @staticmethod
            >>> def backward(ctx, grad_output):
            ...     x, = ctx.saved_tensors
            ...     return grad_output * 2 * x  # Gradient for x**2
        """
        raise NotImplementedError

    @classmethod
    def apply(cls: Type[TFunction], *args: Any, **kwargs: Any) -> Tensor:
        """
        Applies the operation and builds the computational graph.

        This is the main entry point for executing differentiable operations.
        It orchestrates the entire forward pass pipeline:

        1. Creates a fresh Context for this operation
        2. Determines base dtype from inputs for numerical consistency
        3. Converts Tensor inputs to numpy arrays
        4. Tracks which tensors need gradients
        5. Executes forward() with raw arrays
        6. Handles dtype casting and validation
        7. Creates output Tensor with proper grad_fn linkage
        8. Attaches Context and inputs for backward pass

        Args:
            *args: Operation inputs (Tensors, scalars, arrays, etc.)
            **kwargs: Additional keyword arguments for the operation.

        Returns:
            Output Tensor with grad_fn attached if any input requires_grad
            and gradients are enabled globally.

        Notes:
            - Automatically manages computational graph construction
            - Preserves dtype for boolean, integer, and complex outputs
            - Casts floating-point outputs to base_dtype for consistency
            - Only attaches grad_fn when gradients are needed and enabled

        Examples:
            >>> import nova
            >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
            >>> y = Add.apply(x, 3.0)  # Creates node in computation graph
            >>> print(y.grad_fn)  # <AddBackward>
        """
        from nova import Tensor

        # Create context for saving intermediate values
        ctx = Context()

        # Determine dtype for numerical stability across mixed inputs
        base_dtype = determine_base_dtype(args)
        processor = ArgumentProcessor(base_dtype)

        # Convert Tensors to numpy arrays and track gradient requirements
        raw_args, raw_kwargs = processor.process_args(args, kwargs)
        tensors_in_graph = processor.get_tracked_tensors()

        # Execute forward pass with raw numpy arrays
        output = cls.forward(ctx, *raw_args, **raw_kwargs)

        # Ensure output is numpy array
        if not isinstance(output, np.ndarray):
            output = np.array(output)

        output_dtype = output.dtype

        # Cast floating-point outputs to base dtype for consistency
        # Keep boolean, integer, and complex dtypes as-is
        if (
            not np.issubdtype(output_dtype, np.bool_)
            and not np.issubdtype(output_dtype, np.integer)
            and not np.issubdtype(output_dtype, np.complexfloating)
        ):
            output = output.astype(base_dtype, copy=False)
            output_dtype = base_dtype

        # Determine if output needs gradient tracking
        requires_grad = (
            any(t.requires_grad for t in tensors_in_graph) and nova.is_grad_enabled()
        )

        # Create output tensor
        result = Tensor(
            output,
            requires_grad=requires_grad,
            dtype=output_dtype,
            grad_fn=cls if requires_grad else None,
            copy=False,
        )

        # Attach computational graph metadata for backward pass
        if requires_grad:
            result._inputs = tensors_in_graph
            result._ctx = ctx

        return result
