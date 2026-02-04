from __future__ import annotations
from numpy import ndarray
import numpy as np
from typing import TYPE_CHECKING, Optional
from nova.utils import registry_op
from nova.autograd._ops.utils import unbroadcasting
from nova.utils.decorators import no_inplace_op
from nova.autograd.function import Function

if TYPE_CHECKING:
    from nova._typing import Gradients, Dim
    from nova.autograd.engine import Context

__all__ = ["Det", "Diag", "Dot", "Inv", "MatMul", "Norm", "Trace"]


@no_inplace_op
@registry_op("det")
class Det(Function):
    """
    Determinant of a square matrix.

    Forward: out = det(input)
    Backward: ∂L/∂input = ∂L/∂out * det(input) * (input^-T)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the determinant of a square matrix."""
        result = np.linalg.det(input)
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for determinant.

        The gradient is: grad_input = det(input) * grad_output * input^-T
        """
        input, det_val = ctx.saved_tensors
        inv_T = np.linalg.inv(input).T

        grad_input = det_val * grad_output * inv_T
        return (grad_input,)


@no_inplace_op
@registry_op("diag")
class Diag(Function):
    """
    Diagonal of a matrix or vector.

    Forward: out = diag(input)
    Backward: ∂L/∂input = diag(∂L/∂out)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, diagonal: int = 0) -> ndarray:
        """compute the diagonal of a vector or matrix"""
        ctx.diagonal = diagonal
        return np.diag(input, diagonal)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for diagonal

        The gradient is: ∂L/∂input = diag(∂L/∂out)
        """
        diagonal = ctx.diagonal
        return (np.diag(grad_output, diagonal),)


@registry_op("dot")
class Dot(Function):
    """
    Matrix-vector or matrix-matrix dot product.

    Forward: out = input · other
    Backward: ∂L/∂input = grad_output · other^T, ∂L/∂other = input^T · grad_output
    """

    @staticmethod
    def forward(
        ctx: Context, input: ndarray, other: ndarray, _out: Optional[ndarray] = None
    ) -> ndarray:
        """Compute the dot product between input and other."""
        ctx.save_for_backward(input, other)
        return np.dot(input, other, out=_out)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for dot product.

        The gradient is: grad_input = grad_output · other^T, grad_other = input^T · grad_output
        """
        input, other = ctx.saved_tensors

        if input.ndim <= 1 and other.ndim <= 1:
            # Vectores
            grad_input = grad_output * other
            grad_other = grad_output * input
        else:
            # Matrices
            grad_input = np.dot(grad_output, other.T)
            grad_other = np.dot(input.T, grad_output)

        return (grad_input, grad_other)


@no_inplace_op
@registry_op("inv")
class Inv(Function):
    """
    Matrix inversion.

    Forward: out = inv(input)
    Backward: ∂L/∂input = -inv(input)^T · grad_output · inv(input)^T
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the inverse of a square matrix."""
        result = np.linalg.inv(input)
        ctx.save_for_backward(result.T)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for inverse matrix.

        The gradient is: grad_input = -inv(input)^T · grad_output · inv(input)^T
        """
        (inv_T,) = ctx.saved_tensors
        grad_input = -inv_T @ grad_output @ inv_T
        return (grad_input,)


@registry_op("matmul")
class MatMul(Function):
    """
    Matrix multiplication using '@' operator.

    Forward: out = input @ other
    Backward: ∂L/∂input = grad_output @ other^T, ∂L/∂other = input^T @ grad_output
    """

    @staticmethod
    def forward(
        ctx: Context, input: ndarray, other: ndarray, _out: Optional[ndarray] = None
    ) -> ndarray:
        """Compute matrix multiplication of input and other."""
        ctx.save_for_backward(input, other)
        return np.matmul(input, other, out=_out)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for matmul.

        The gradient is: grad_input = grad_output @ other^T, grad_other = input^T @ grad_output
        """
        input, other = ctx.saved_tensors
        grad_input = grad_output @ other.T
        grad_other = input.T @ grad_output
        return (grad_input, grad_other)


@no_inplace_op
@registry_op("norm")
class Norm(Function):
    """
    Vector or matrix norm.

    Forward: computes np.linalg.norm(input, ord=ord, axis=dim, keepdims=keepdims)
    Backward: ∂L/∂input = grad_output * input / ||input||
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        ord: int = 2,
        dim: Optional[Dim] = None,
        keepdims: bool = False,
    ) -> ndarray:
        """Compute the norm of input along given axis/dim."""
        result = np.linalg.norm(input, ord=ord, axis=dim, keepdims=keepdims)
        ctx.ord = ord
        ctx.dim = dim
        ctx.keepdims = keepdims
        ctx.save_for_backward(input, result)
        ctx.saved_shapes = input.shape
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for norm.

        The gradient is: grad_input = grad_output * input / ||input||
        """
        input, out = ctx.saved_tensors
        input_shape = ctx.saved_shapes

        actual_out = out
        actual_grad = grad_output

        if not ctx.keepdims and input.ndim > (
            out.ndim if np.isscalar(out) else out.ndim
        ):
            target_shape = list(input.shape)
            if ctx.dim is not None:
                dims = [ctx.dim] if isinstance(ctx.dim, int) else ctx.dim
                for d in dims:
                    target_shape[d] = 1
            else:
                target_shape = [1] * input.ndim

            actual_out = out.reshape(target_shape)
            actual_grad = grad_output.reshape(target_shape)

        safe_out = np.where(actual_out > 0, actual_out, 1.0)

        if ctx.ord == 2:
            grad_input = actual_grad * (input / safe_out)
        elif ctx.ord == 1:
            grad_input = actual_grad * np.sign(input)
        elif ctx.ord == np.inf:
            mask = np.abs(input) == actual_out
            grad_input = actual_grad * mask * np.sign(input)
        else:
            term1 = input * np.power(np.abs(input), ord - 2.0)
            term2 = np.power(safe_out, ord - 1.0)
            grad_input = actual_grad * (term1 / term2)

        return (unbroadcasting(grad_input, input_shape),)


@no_inplace_op
@registry_op("trace")
class Trace(Function):
    """
    Trace of a square matrix.

    Forward: out = sum(diag(input))
    Backward: ∂L/∂input = grad_output * I
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Compute the trace (sum of diagonal elements) of a square matrix."""
        ctx.save_for_backward(input)
        return np.linalg.trace(input)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for trace.

        The gradient is: grad_input = grad_output * identity matrix
        """
        (input,) = ctx.saved_tensors
        N = input.shape[0]
        grad_input = np.eye(N) * grad_output
        return (grad_input,)
