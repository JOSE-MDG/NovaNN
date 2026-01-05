from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients, Dim


def _normalize_dim(dim: Optional[Dim] = None) -> Optional[tuple[int, ...]]:
    if dim is None:
        return None
    elif isinstance(dim, int):
        return (dim,)
    elif isinstance(dim, (tuple, list)):
        return tuple(dim)
    else:
        raise TypeError(f"Invalid dim type {type(dim)}")


@registry_op("sum")
class Sum(Function):
    """
    Sum of tensor elements.

    Forward: out = sum(input)
    Backward: ∂L/∂input = broadcast(grad_output)
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        dim: Optional[Dim] = None,
        keepdims: bool = False,
    ) -> ndarray:
        """Compute the sum over specified dimensions."""
        dim = _normalize_dim(dim)

        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.saved_shape = input.shape

        return np.sum(input, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for sum.

        The gradient is: grad_output broadcasted to input shape
        """
        if not ctx.keepdims and ctx.dim is not None:
            grad_output = np.expand_dims(grad_output, ctx.dim)

        grad_input = np.broadcast_to(grad_output, ctx.saved_shape)
        return (grad_input,)


@registry_op("mean")
class Mean(Function):
    """
    Mean of tensor elements.

    Forward: out = mean(input)
    Backward: ∂L/∂input = broadcast(grad_output) / N
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        dim: Optional[Dim] = None,
        keepdims: bool = False,
    ) -> ndarray:
        """Compute the mean over specified dimensions."""
        dim = _normalize_dim(dim)

        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.saved_shape = input.shape

        if dim is None:
            ctx.N = input.size
        else:
            ctx.N = 1
            for d in dim:
                ctx.N *= input.shape[d]

        return np.mean(input, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for mean.

        The gradient is: broadcast(grad_output) divided by number of elements
        """
        if not ctx.keepdims and ctx.dim is not None:
            grad_output = np.expand_dims(grad_output, ctx.dim)

        grad_input = np.broadcast_to(grad_output, ctx.saved_shape) / ctx.N
        return (grad_input,)


@registry_op("var")
class Var(Function):
    """
    Variance of tensor elements.

    Forward: out = var(input)
    Backward: ∂L/∂input = (2 / N) * (input - mean) * grad_output
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        dim: Optional[Dim] = None,
        keepdims: bool = False,
    ) -> ndarray:
        """Compute the variance over specified dimensions."""
        dim = _normalize_dim(dim)

        ctx.save_for_backward(input)
        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.saved_shape = input.shape

        if dim is None:
            ctx.N = input.size
        else:
            ctx.N = 1
            for d in dim:
                ctx.N *= input.shape[d]

        mean_val = np.mean(input, axis=dim, keepdims=keepdims)
        diff = input - mean_val

        ctx.diff = diff
        return np.mean(diff * diff, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for variance.

        The gradient is: (2 / N) * (input - mean) * grad_output
        """
        if not ctx.keepdims and ctx.dim is not None:
            grad_output = np.expand_dims(grad_output, ctx.dim)

        grad_output = np.broadcast_to(grad_output, ctx.saved_shape)
        grad_input = (2.0 / ctx.N) * ctx.diff * grad_output
        return (grad_input,)


@registry_op("max")
class Max(Function):
    """
    Maximum value of tensor elements.

    Forward: out = max(input)
    Backward: ∂L/∂input = grad_output distributed over max elements
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        dim: Optional[Dim] = None,
        keepdims: bool = False,
    ) -> ndarray:
        """Compute the maximum over specified dimensions."""
        dim = _normalize_dim(dim)

        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.saved_shape = input.shape

        if dim is None:
            ctx.mask = input == input.max()
        else:
            ctx.mask = input == input.max(axis=dim, keepdims=True)

        return np.max(input, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for max.

        The gradient is: grad_output evenly distributed among max elements
        """
        if not ctx.keepdims and ctx.dim is not None:
            grad_output = np.expand_dims(grad_output, ctx.dim)

        grad_output = np.broadcast_to(grad_output, ctx.saved_shape)

        grad_input = grad_output * ctx.mask
        grad_input /= ctx.mask.sum(axis=ctx.dim, keepdims=True)
        return (grad_input,)


@registry_op("min")
class Min(Function):
    """
    Minimum value of tensor elements.

    Forward: out = min(input)
    Backward: ∂L/∂input = grad_output distributed over min elements
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        dim: Optional[Dim] = None,
        keepdims: bool = False,
    ) -> ndarray:
        """Compute the minimum over specified dimensions."""
        dim = _normalize_dim(dim)

        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.saved_shape = input.shape

        if dim is None:
            ctx.mask = input == input.min()
        else:
            ctx.mask = input == input.min(axis=dim, keepdims=True)

        return np.min(input, axis=dim, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for min.

        The gradient is: grad_output evenly distributed among min elements
        """
        if not ctx.keepdims and ctx.dim is not None:
            grad_output = np.expand_dims(grad_output, ctx.dim)

        grad_output = np.broadcast_to(grad_output, ctx.saved_shape)

        grad_input = grad_output * ctx.mask
        grad_input /= ctx.mask.sum(axis=ctx.dim, keepdims=True)
        return (grad_input,)
