from __future__ import annotations
import numpy as np
import nova
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nova._typing import Dtype


class Generator:
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed=seed)

    def manual_seed(self, seed: int):
        self._rng = np.random.default_rng(seed=seed)


_default_generator = Generator()


def manual_seed(seed: Optional[int] = None) -> None:
    _default_generator.manual_seed(seed)


def rand(
    *size: Optional[int],
    dtype: Dtype | None = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> nova.Tensor:

    gen = generator or _default_generator
    if size is None:

        data = gen._rng.random(dtype=dtype)

    data = gen._rng.random(size, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def randn(
    *size: Optional[int],
    dtype: Dtype | None = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> nova.Tensor:

    gen = generator or _default_generator

    if size is None:
        data = gen._rng.standard_normal(dtype=dtype)

    data = gen._rng.standard_normal(size, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def randint(
    low: int,
    high: int | None,
    size: Optional[tuple[int, ...]],
    *,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> nova.Tensor:

    gen = generator or _default_generator
    data = gen._rng.integers(
        low, high, size, dtype=dtype if dtype is not None else nova.long
    )
    return nova.Tensor(
        data,
        requires_grad=requires_grad,
        dtype=dtype if dtype is not None else nova.long,
    )


def randperm(
    n: int,
    *,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> nova.Tensor:

    gen = generator or _default_generator
    data = gen._rng.permutation(n)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def normal(
    mean: float,
    std: float,
    size: Optional[tuple[int, ...]],
    *,
    generator: Optional[Generator] = None,
) -> nova.Tensor:

    gen = generator or _default_generator
    data = gen._rng.normal(mean, std, size)
    return nova.Tensor(data)


def uniform(
    low: float,
    high: float,
    size: Optional[tuple[int, ...]],
    *,
    generator: Optional[Generator] = None,
) -> nova.Tensor:

    gen = generator or _default_generator
    data = gen._rng.uniform(low, high, size)

    return nova.Tensor(data)
