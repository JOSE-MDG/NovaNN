from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients, Dim


@registry_op("permute")
class Permute(Function):
    """
    Permute the dimensions of a tensor.

    Forward: out = transpose(input, dims)
    Backward: ∂L/∂input = transpose(grad_output, inverse(dims))
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, *dims: Optional[Dim]) -> ndarray:
        """Reorder tensor dimensions according to dims."""
        if not dims:
            dims = None
        ctx.dims = dims
        return np.transpose(input, axes=dims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for permute.

        The gradient is: transpose(grad_output, inverse permutation)
        """
        if ctx.dims is None:
            return (grad_output.T,)

        inv_dims = np.argsort(ctx.dims)
        grad_input = np.transpose(grad_output, inv_dims)
        return (grad_input,)


@registry_op("reshape")
class Reshape(Function):
    """
    Reshape a tensor without changing its data.

    Forward: out = reshape(input, new_shape)
    Backward: ∂L/∂input = reshape(grad_output, original_shape)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, *size: Dim) -> ndarray:
        """Reshape tensor to the given size."""
        ctx.saved_shape = input.shape
        return np.reshape(input, shape=size)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for reshape.

        The gradient is: reshape grad_output to original input shape
        """
        return (grad_output.reshape(ctx.saved_shape),)


@registry_op("squeeze")
class Squeeze(Function):
    """
    Remove dimensions of size 1.

    Forward: out = squeeze(input)
    Backward: ∂L/∂input = reshape(grad_output, original_shape)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, dim: Optional[Dim] = None) -> ndarray:
        """Remove singleton dimensions."""
        ctx.saved_shape = input.shape
        return np.squeeze(input, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for squeeze.

        The gradient is: reshape grad_output to original input shape
        """
        return (grad_output.reshape(ctx.saved_shape),)


@registry_op("unsqueeze")
class UnSqueeze(Function):
    """
    Insert a dimension of size 1.

    Forward: out = expand_dims(input)
    Backward: ∂L/∂input = reshape(grad_output, original_shape)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, dim: Dim) -> ndarray:
        """Insert a singleton dimension."""
        ctx.saved_shape = input.shape
        return np.expand_dims(input, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for unsqueeze.

        The gradient is: reshape grad_output to original input shape
        """
        return (grad_output.reshape(ctx.saved_shape),)


@registry_op("stack")
class Stack(Function):
    """
    Stack tensors along a new dimension.

    Forward: out = stack(inputs, dim)
    Backward: ∂L/∂inputs = split(grad_output)
    """

    @staticmethod
    def forward(ctx: Context, inputs: list[ndarray], dim: Dim = 0) -> ndarray:
        """Stack tensors along a new axis."""
        ctx.dim = dim
        ctx.N = len(inputs)
        return np.stack(inputs, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for stack.

        The gradient is: split grad_output along stacked dimension
        """
        grads = np.split(grad_output, ctx.N, axis=ctx.dim)
        grads = [g.squeeze(ctx.dim) for g in grads]
        return (*grads,)


@registry_op("concat")
class Concat(Function):
    """
    Concatenate tensors along an existing dimension.

    Forward: out = concatenate(tensors, dim)
    Backward: ∂L/∂tensors = split(grad_output)
    """

    @staticmethod
    def forward(ctx: Context, tensors: list[ndarray], dim: Dim = 0) -> ndarray:
        """Concatenate tensors along a dimension."""
        ctx.saved_shapes = [t.shape for t in tensors]
        ctx.dim = dim
        return np.concatenate(tensors, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for concat.

        The gradient is: split grad_output according to original tensor sizes
        """
        sizes = [s[ctx.dim] for s in ctx.saved_shapes]
        offsets = np.cumsum(sizes)[:-1]
        grads = np.split(grad_output, offsets, axis=ctx.dim)
        return (*grads,)


@registry_op("split")
class Split(Function):
    """
    Split a tensor into multiple sections.

    Forward: out = split(input, sections)
    Backward: ∂L/∂input = concatenate(grad_outputs)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, sections: int, dim: Dim = 0) -> ndarray:
        """Split tensor into equal sections."""
        ctx.dim = dim
        return np.array_split(input, sections, dim)

    @staticmethod
    def backward(ctx: Context, *grad_output: ndarray) -> Gradients:
        """
        Backward pass for split.

        The gradient is: concatenate all grad_outputs
        """
        grad_input = np.concatenate(grad_output, axis=ctx.dim)
        return (grad_input,)


@registry_op("clamp")
class Clamp(Function):
    """
    Clamp tensor values to a given range.

    Forward: out = clip(input, min, max)
    Backward: ∂L/∂input = grad_output where input ∈ [min, max]
    """

    @staticmethod
    def forward(
        ctx: Context, input: ndarray, min_val: float, max_val: float
    ) -> ndarray:
        """Clamp tensor values to a range."""
        ctx.save_for_backward(input)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return np.clip(input, min_val, max_val)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for clamp.

        The gradient is: grad_output masked by clamp range
        """
        (input,) = ctx.saved_tensors
        mask = (input >= ctx.min_val) & (input <= ctx.max_val)
        grad_input = grad_output * mask
        return (grad_input,)


@registry_op("pad")
class Pad(Function):
    """
    Pad a tensor.

    Forward: out = pad(input)
    Backward: ∂L/∂input = slice(grad_output)
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        pad_width: tuple[tuple[int, ...], ...],
        mode: str = "constant",
    ) -> ndarray:
        """Pad tensor according to pad_width."""
        ctx.pad_width = pad_width
        return np.pad(input, pad_width, mode=mode)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for pad.

        The gradient is: remove padded regions from grad_output
        """
        slices = []
        for i, (before, after) in enumerate(ctx.pad_width):
            end = grad_output.shape[i] - after
            slices.append(slice(before, end))

        grad_input = grad_output[tuple(slices)]
        return (grad_input,)
