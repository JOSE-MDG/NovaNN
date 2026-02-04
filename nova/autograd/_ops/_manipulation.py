from __future__ import annotations
import numpy as np
from numpy import ndarray
from nova.autograd.function import Function
from nova.utils import registry_op
from nova.utils.decorators import no_inplace_op
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova.autograd.engine import Context
    from nova._typing import Gradients, Dim

__all__ = [
    "Permute",
    "Reshape",
    "Squeeze",
    "UnSqueeze",
    "Stack",
    "Concat",
    "Split",
    "Clone",
    "Tile",
    "Repeat",
    "Pad",
]


@no_inplace_op
@registry_op("permute")
class Permute(Function):
    """
    Permute the dimensions of a tensor.

    Forward: out = permute(input, dims)
    Backward: ∂L/∂input = permute(∂L/∂out, inverse_dims)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, *dims: Optional[Dim]) -> ndarray:
        """Permute tensor dimensions according to dims"""
        if not dims:
            dims = None
        ctx.dims = dims
        return np.transpose(input, axes=dims)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for permute

        The gradient is: ∂L/∂input = permute(∂L/∂out, inverse_permutation)
        """
        if ctx.dims is None:
            return (grad_output.T, None)

        inv_dims = np.argsort(ctx.dims)
        grad_input = np.transpose(grad_output, inv_dims)
        return (grad_input,)


@no_inplace_op
@registry_op("reshape")
class Reshape(Function):
    """
    Reshape a tensor to a new shape.

    Forward: out = reshape(input, size)
    Backward: ∂L/∂input = reshape(∂L/∂out, original_shape)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, *size: Dim) -> ndarray:
        """Reshape tensor to new shape"""
        ctx.saved_shapes = input.shape
        return np.reshape(input, shape=size)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for reshape

        The gradient is: ∂L/∂input = reshape(∂L/∂out, original_shape)
        """
        input_shape = ctx.saved_shapes
        return (grad_output.reshape(input_shape),)


@no_inplace_op
@registry_op("squeeze")
class Squeeze(Function):
    """
    Remove dimensions of size 1 from a tensor.

    Forward: out = squeeze(input, dim)
    Backward: ∂L/∂input = unsqueeze(∂L/∂out, dim)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, dim: Optional[Dim] = None) -> ndarray:
        """Remove dimensions of size 1"""
        ctx.saved_shapes = input.shape
        return np.squeeze(input, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for squeeze

        The gradient is: ∂L/∂input = reshape(∂L/∂out, original_shape)
        """
        input_shape = ctx.saved_shapes
        return (grad_output.reshape(input_shape),)


@no_inplace_op
@registry_op("unsqueeze")
class UnSqueeze(Function):
    """
    Add a dimension of size 1 to a tensor.

    Forward: out = unsqueeze(input, dim)
    Backward: ∂L/∂input = squeeze(∂L/∂out, dim)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, dim: Dim) -> ndarray:
        """Add a dimension of size 1 at position dim"""
        ctx.saved_shapes = input.shape
        return np.expand_dims(input, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for unsqueeze

        The gradient is: ∂L/∂input = reshape(∂L/∂out, original_shape)
        """
        input_shape = ctx.saved_shapes
        return (grad_output.reshape(input_shape),)


@no_inplace_op
@registry_op("stack")
class Stack(Function):
    """
    Stack tensors along a new dimension.

    Forward: out = stack([t1, t2, ...], dim)
    Backward: ∂L/∂ti = split(∂L/∂out, dim)[i]
    """

    @staticmethod
    def forward(
        ctx: Context,
        inputs: list[ndarray],
        dim: Dim = 0,
    ) -> ndarray:
        """Stack list of tensors along new dimension"""
        ctx.dim = dim
        ctx.N = len(inputs)
        return np.stack(inputs, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for stack

        The gradient is: ∂L/∂inputs[i] = split(∂L/∂out, dim)[i] (squeezed)
        """
        grad_inputs = np.split(grad_output, ctx.N, axis=ctx.dim)
        grad_inputs = [g.squeeze(ctx.dim) for g in grad_inputs]
        return (*grad_inputs, None)


@no_inplace_op
@registry_op("concat")
class Concat(Function):
    """
    Concatenate tensors along an existing dimension.

    Forward: out = concat([t1, t2, ...], dim)
    Backward: ∂L/∂ti = split(∂L/∂out, sizes, dim)[i]
    """

    @staticmethod
    def forward(ctx: Context, inputs: list[ndarray], dim: ndarray = 0) -> ndarray:
        """Concatenate tensors along existing dimension"""
        ctx.saved_shapes = [i.shape for i in inputs]
        ctx.dim = dim
        return np.concatenate(inputs, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for concat

        The gradient is: ∂L/∂inputs[i] = split(∂L/∂out, original_sizes, dim)[i]
        """
        shapes = ctx.saved_shapes
        sizes = [s[ctx.dim] for s in shapes]
        offsets = np.cumsum(sizes)[:-1]
        grad_inputs = np.split(grad_output, offsets, axis=ctx.dim)
        return (*grad_inputs, None)


@no_inplace_op
@registry_op("split")
class Split(Function):
    """
    Split a tensor into multiple chunks along a dimension.

    Forward: out = split(input, sections, dim)
    Backward: ∂L/∂input = concat([∂L/∂out[0], ∂L/∂out[1], ...], dim)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, sections: int, dim: Dim = 0) -> ndarray:
        """Split tensor into sections along dimension"""
        ctx.dim = dim
        return np.array_split(input, sections, dim)

    @staticmethod
    def backward(ctx: Context, *grad_output: ndarray) -> Gradients:
        """
        Backward pass for split

        The gradient is: ∂L/∂input = concat(∂L/∂outputs, dim)
        """
        grad_input = np.concatenate(grad_output, ctx.dim)
        return (grad_input,)


@no_inplace_op
@registry_op("tile")
class Tile(Function):
    """
    Tile a tensor by repeating it along each dimension.

    Forward: out = tile(input, repeats)
    Backward: ∂L/∂input = sum(∂L/∂out, over repeated blocks)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, repeats: int) -> ndarray:
        """Tile tensor by repeating along each dimension"""
        ctx.repeats = repeats
        ctx.saved_shapes = input.shape
        return np.tile(input, repeats)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for tile

        The gradient is: ∂L/∂input = sum of gradients over all repeated blocks
        """
        shape_input = ctx.saved_shapes
        grad_input = grad_output
        for axis, rep in enumerate(ctx.repeats):
            grad_input = grad_input.reshape(
                rep, shape_input[axis], *shape_input[axis + 1 :]
            ).sum(axis=0)
        return (grad_input,)


@no_inplace_op
@registry_op("repeat")
class Repeat(Function):
    """
    Repeat elements of a tensor along a dimension.

    Forward: out = repeat(input, repeats, dim)
    Backward: ∂L/∂input = sum(∂L/∂out, over repeated elements)
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray, repeats: int, dim: Dim = 0) -> ndarray:
        """Repeat tensor elements along dimension"""
        ctx.saved_shapes = input.shape
        ctx.repeats = repeats
        ctx.dim = dim
        return np.repeat(input, repeats, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output) -> Gradients:
        """
        Backward pass for repeat

        The gradient is: ∂L/∂input = sum of gradients over all repeated elements
        """
        shape_input = ctx.saved_shapes

        if ctx.dim is None:
            grad_input = grad_output.reshape(-1, ctx.repeats).sum(axis=1)
            grad_input = grad_input.reshape(shape_input)
        else:
            grad_input = grad_output.reshape(
                *shape_input[: ctx.dim],
                shape_input[ctx.dim],
                ctx.repeats,
                *shape_input[ctx.dim + 1 :],
            ).sum(axis=ctx.dim + 1)

        return (grad_input,)


@no_inplace_op
@registry_op("pad")
class Pad(Function):
    """
    Pad a tensor with values along its dimensions.

    Forward: out = pad(input, pad_width, mode)
    Backward: ∂L/∂input = unpad(∂L/∂out, pad_width)
    """

    @staticmethod
    def forward(
        ctx: Context,
        input: ndarray,
        pad_width: tuple[tuple[int, ...], ...],
        mode: str = "constant",
    ) -> ndarray:
        """Pad tensor with specified padding width and mode"""
        ctx.pad_width = pad_width
        return np.pad(input, pad_width, mode=mode)

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for pad

        The gradient is: ∂L/∂input = slice(∂L/∂out) removing padded regions
        """
        pad_width = ctx.pad_width
        slices = []
        for i, (before, after) in enumerate(pad_width):
            end = grad_output.shape[i] - after
            slices.append(slice(before, end))
        grad_input = grad_output[tuple(slices)]
        return (grad_input,)


@no_inplace_op
@registry_op("clone")
class Clone(Function):
    """
    Clone a tensor.

    Forward: out = copy(input)
    Backward: ∂L/∂input = grad_output
    """

    @staticmethod
    def forward(ctx: Context, input: ndarray) -> ndarray:
        """Return a copy of the input tensor."""
        return input.copy()

    @staticmethod
    def backward(ctx: Context, grad_output: ndarray) -> Gradients:
        """
        Backward pass for clone.

        The gradient is: grad_output
        """
        return (grad_output,)
