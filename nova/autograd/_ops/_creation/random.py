from __future__ import annotations
import numpy as np
import nova
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nova._typing import Dim, Dtype


def manual_seed(seed: Optional[int] = None) -> None:
    np.random.seed(seed=seed)


def rand(
    *size: Optional[int], dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    if size is None:
        data = np.random.rand()

    data = np.random.rand(*size)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def randn(
    *size: Optional[int], dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    if size is None:
        data = np.random.rand()

    data = np.random.randn(*size)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def randint(
    low: int,
    high: int | None,
    size: Optional[tuple[int, ...]],
    *,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
) -> nova.Tensor:

    data = np.random.randint(low, high, size)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def randperm(
    n: int, *, dtype: Dtype | None = None, requires_grad: bool = False
) -> nova.Tensor:
    data = np.random.permutation(n)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def normal(
    mean: float, std: float, size: Optional[tuple[int, ...]], dtype: Dtype | None = None
) -> nova.Tensor:
    data = np.random.normal(mean, std, size)
    return nova.Tensor(data, dtype=dtype)


def uniform(
    low: float, high: float, size: Optional[tuple[int, ...]], dtype: Dtype | None = None
) -> nova.Tensor:
    data = np.random.uniform(low, high, size)

    return nova.Tensor(data, dtype=dtype)
