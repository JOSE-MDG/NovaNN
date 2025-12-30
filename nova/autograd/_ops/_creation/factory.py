from __future__ import annotations
import nova
import numpy as np
from typing import Any, Optional, TYPE_CHECKING, Literal
from nova.utils import ensure_tensor

if TYPE_CHECKING:
    from nova._typing import Dim, Dtype


def sqrt(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.sqrt()


def mean(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.mean(dim, keepdims)


def var(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    input = ensure_tensor(input)
    diff = input - input.mean(dim, keepdims)
    diff_sq = diff**2
    var = mean(diff_sq, dim, keepdims)
    return var


def std(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    input = ensure_tensor(input)

    return sqrt(var(input, dim, keepdims)).to(input.dtype)


def empty(
    size: tuple[int, ...], *, dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    data = np.empty(shape=size, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def min(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    input = ensure_tensor(input)

    return input.min(dim=dim, keepdims=keepdims)


def max(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    input = ensure_tensor(input)

    return input.max(dim=dim, keepdims=keepdims)


def sum(input: nova.Tensor, dim: Dim = None, keepdims: bool = False) -> nova.Tensor:
    input = ensure_tensor(input)

    return input.sum(dim=dim, keepdims=keepdims)


def pow(input: nova.Tensor, exponent: nova.Tensor | int | float) -> nova.Tensor:
    input = ensure_tensor(input)
    exponent = ensure_tensor(exponent)

    return input**exponent


def maximum(input: nova.Tensor, other: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    other = ensure_tensor(other)

    return input.maximum(other)


def minimum(input: nova.Tensor, other: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    other = ensure_tensor(other)

    return input.minimum(other)


def clamp(input: nova.Tensor, min_val: float, max_val: float) -> nova.Tensor:
    input = ensure_tensor(input)

    return input.clamp(min_val, max_val)


def split(input: nova.Tensor, sections: int, dim: Dim = None) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.split(sections=sections, dim=dim)


def tile(input: nova.Tensor, repeats: int) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.tile(repeats)


def repeat_interleave(input: nova.Tensor, repeats: int, dim):
    input = ensure_tensor(input)
    return input.repeat(repeats=repeats, dim=dim)


def pad(
    input: nova.Tensor,
    pad_width: tuple[tuple[int, ...], ...] | tuple[int, ...],
    mode: Literal["constant", "reflect", "wrap", "edge"] = "constant",
):
    input = ensure_tensor(input)
    return input.pad(pad_width, mode)


def floor(input: nova.Tensor):
    input = ensure_tensor(input)
    return input.floor()


def exp(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.exp()


def sin(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.sin()


def cos(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.cos()


def tan(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.tan()


def tanh(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.tanh()


def sec(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.sec()


def cot(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.cot()


def log(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.log()


def reshape(input: nova.Tensor, dims: Dim) -> nova.Tensor:
    input = ensure_tensor(input)

    return input.reshape(*dims)


def dot(input: nova.Tensor, other: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    other = ensure_tensor(other)

    return input.dot(other)


def det(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.det()


def inv(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)

    return input.inv()


def trace(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)

    return input.trace()


def norm(
    input: nova.Tensor, ord: int = 2, dim: Optional[Dim] = None, keepdims: bool = False
) -> nova.Tensor:
    input = ensure_tensor(input)

    return input.norm(ord, dim=dim, keepdims=keepdims)


def abs(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.abs()


def sign(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)

    return input.sign()


def arcsin(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.arcsin()


def arccos(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.arccos()


def arctan(input: nova.Tensor) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.arctan()


def where(condition: nova.Tensor, x: nova.Tensor, y: nova.Tensor):
    from nova.autograd._ops import Where

    x = ensure_tensor(x)
    y = ensure_tensor(y)
    return Where.apply(condition, x, y)


def permute(input: nova.Tensor, *dims: Dim) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.permute(dims)


def unsqueeze(input: nova.Tensor, dim: Dim) -> nova.Tensor:
    input = ensure_tensor(input)
    return input.unsqueeze(dim)


def cat(inputs: list[nova.Tensor], dim: Dim = None):
    from nova.autograd._ops import Concat

    return Concat.apply(inputs, dim)


def eye(
    N: int, M: Optional[int] = None, K: int = 0, requires_grad: bool = False
) -> nova.Tensor:
    if M is None:
        M = N

    data = np.eye(N, M, k=K)
    return nova.Tensor(data, dtype=data.dtype, requires_grad=requires_grad)


def one_hot(
    labels: nova.Tensor, num_classes: int, requires_grad: bool = False
) -> nova.Tensor:
    labels = ensure_tensor(labels)
    data = np.eye(num_classes)[labels.data.astype(nova.long)]
    return nova.Tensor(data, dtype=nova.long, requires_grad=requires_grad)


def full(
    size: tuple[int, ...] | list,
    fill_value: Any,
    *,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
) -> nova.Tensor:
    data = np.full(shape=size, fill_value=fill_value, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def arange(
    start: int | float,
    stop: Optional[int | float] = None,
    step: int | float = 1,
    *,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
) -> nova.Tensor:
    if stop is None:
        stop = start
        start = 0

    data = np.arange(start=start, stop=stop, step=step, dtype=dtype)

    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def unique(
    input: nova.Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: Dim = None,
):
    input = ensure_tensor(input)

    unique = np.unique(
        input.data,
        sorted=sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=dim,
    )

    return nova.Tensor(unique, dtype=unique.dtype, requires_grad=False)


def argmin(input: nova.Tensor, dim: Dim = None, keepdims: bool = False):
    input = ensure_tensor(input)

    return input.argmin(dim=dim, keepdims=keepdims)


def argmax(input: nova.Tensor, dim: Dim = None, keepdims: bool = False):
    input = ensure_tensor(input)

    return input.argmax(dim=dim, keepdims=keepdims)


def argsort(input: nova.Tensor, dim: Dim = -1, kind=None, order=None):
    input = ensure_tensor(input)

    return input.argsort(dim=dim, kind=kind, order=order)


def argwhere(input: nova.Tensor):
    input = ensure_tensor(input)

    return input.argwhere()


def stack(inputs: list[nova.Tensor], dim: Dim = 0):
    from nova.autograd._ops import Stack

    return Stack.apply(inputs, dim)


def zeros(
    size: tuple[int, ...], dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    data = np.zeros(shape=size, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros_like(
    input: nova.Tensor, dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    input = ensure_tensor(input)
    data = np.zeros_like(input.data, dtype=input.dtype if dtype is None else dtype)
    return nova.Tensor(
        data,
        requires_grad=requires_grad,
        dtype=input.dtype if dtype is None else dtype,
    )


def ones(
    size: tuple[int, ...], dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    data = np.ones(shape=size, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def ones_like(
    input: nova.Tensor, dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    input = ensure_tensor(input)
    data = np.ones_like(input.data, dtype=input.dtype if dtype is None else dtype)
    return nova.Tensor(
        data,
        requires_grad=requires_grad,
        dtype=input.dtype if dtype is None else dtype,
    )


def as_strided(input: nova.Tensor, size: tuple[int, ...], strides: tuple[int, ...]):
    input = ensure_tensor(input)

    return input.as_strided(size=size, strides=strides)


def linspace(
    start: int | float, stop: int | float, num: int | float, requires_grad: bool = False
):
    data = np.linspace(start=start, stop=stop, num=num)
    return nova.Tensor(data, requires_grad=requires_grad)


def any(input: nova.Tensor, dim: Dim = None, keepdims: bool = False):
    input = ensure_tensor(input)
    return input.any(dim, keepdims=keepdims)


def all(input: nova.Tensor, dim: Dim = None, keepdims: bool = False):
    input = ensure_tensor(input)
    return input.all(dim, keepdims=keepdims)


def allclose(
    input: nova.Tensor,
    other: nova.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> bool:
    input = ensure_tensor(input)
    other = ensure_tensor(other)

    return input.allclose(other=other, rtol=rtol, atol=atol, equal_nan=equal_nan)
