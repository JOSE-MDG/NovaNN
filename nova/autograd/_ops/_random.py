from __future__ import annotations
import numpy as np
import nova
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nova._typing import Dtype

__all__ = [
    "Generator",
    "rand",
    "randn",
    "randint",
    "randperm",
    "normal",
    "uniform",
    "manual_seed",
]


class Generator:
    """
    Random number generator wrapper around numpy's Generator.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the generator.

        Args:
            seed (Optional[int]): Seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed=seed)

    def manual_seed(self, seed: int):
        """
        Sets a manual seed for the generator.

        Args:
            seed (int): Seed value.
        """
        self._rng = np.random.default_rng(seed=seed)


_default_generator = Generator()


def manual_seed(seed: Optional[int] = None) -> None:
    """
    Sets the seed of the default generator.

    Args:
        seed (Optional[int]): Seed value.

    Examples:
        >>> nova.manual_seed(42)
    """
    _default_generator.manual_seed(seed)


def rand(
    *size: int,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> nova.Tensor:
    """
    Returns a tensor with random values from a uniform distribution [0, 1).

    Args:
        *size (int): Shape of the output tensor.
        dtype (Optional[Dtype]): Data type of the tensor.
        requires_grad (bool): Whether to track gradients.
        generator (Optional[Generator]): Random number generator.

    Returns:
        nova.Tensor: Random tensor.

    Examples:
        >>> x = nova.rand(2,3)
        >>> print(x.shape)
        (2, 3)
    """
    gen = generator or _default_generator
    if not size:
        size = ()
    dtype = dtype if dtype is not None else nova.float32
    data = gen._rng.random(size=size, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def randn(
    *size: int,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> nova.Tensor:
    """
    Returns a tensor with random values from a standard normal distribution.

    Args:
        *size (int): Shape of the output tensor.
        dtype (Optional[Dtype]): Data type of the tensor.
        requires_grad (bool): Whether to track gradients.
        generator (Optional[Generator]): Random number generator.

    Returns:
        nova.Tensor: Random tensor.

    Examples:
        >>> x = nova.randn(2,3)
        >>> print(x.shape)
        (2, 3)
    """
    gen = generator or _default_generator
    if not size:
        size = ()
    dtype = dtype if dtype is not None else nova.float32
    data = gen._rng.standard_normal(size=size).astype(dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def randint(
    low: int,
    high: Optional[int] = None,
    size: Optional[tuple[int, ...]] = None,
    *,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> nova.Tensor:
    """
    Returns a tensor with random integers in the range [low, high).

    Args:
        low (int): Lower bound (inclusive) if high is specified, else upper bound.
        high (Optional[int]): Upper bound (exclusive). If None, [0, low) is used.
        size (Optional[tuple[int, ...]]): Output shape.
        dtype (Optional[Dtype]): Data type of the tensor.
        requires_grad (bool): Whether to track gradients.
        generator (Optional[Generator]): Random number generator.

    Returns:
        nova.Tensor: Random integer tensor.

    Examples:
        >>> x = nova.randint(0, 5, size=(2,2))
        >>> print(x)
        tensor([[0, 3],
                [1, 4]])
    """
    gen = generator or _default_generator
    dtype = dtype if dtype is not None else nova.long
    if high is None:
        high = low
        low = 0
    data = gen._rng.integers(low, high, size=size, dtype=dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def randperm(
    n: int,
    *,
    dtype: Dtype | None = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None,
) -> nova.Tensor:
    """
    Returns a tensor with a random permutation of integers from 0 to n-1.

    Args:
        n (int): Upper bound (exclusive).
        dtype (Optional[Dtype]): Data type of the tensor.
        requires_grad (bool): Whether to track gradients.
        generator (Optional[Generator]): Random number generator.

    Returns:
        nova.Tensor: Random permutation tensor.

    Examples:
        >>> x = nova.randperm(5)
        >>> print(x)
        tensor([3, 1, 4, 0, 2])
    """
    gen = generator or _default_generator
    dtype = dtype if dtype is not None else nova.long
    data = gen._rng.permutation(n).astype(dtype)
    return nova.Tensor(data, requires_grad=requires_grad, dtype=dtype)


def normal(
    mean: float,
    std: float,
    size: Optional[tuple[int, ...]] = None,
    *,
    generator: Optional[Generator] = None,
) -> nova.Tensor:
    """
    Returns a tensor with random values from a normal distribution with given mean and std.

    Args:
        mean (float): Mean of the distribution.
        std (float): Standard deviation.
        size (Optional[tuple[int, ...]]): Shape of the output tensor.
        generator (Optional[Generator]): Random number generator.

    Returns:
        nova.Tensor: Random tensor from normal distribution.

    Examples:
        >>> x = nova.normal(0, 1, size=(2,2))
        >>> print(x)
    """
    gen = generator or _default_generator
    if size is None:
        size = ()
    data = gen._rng.normal(loc=mean, scale=std, size=size)
    return nova.Tensor(data)


def uniform(
    low: float,
    high: float,
    size: Optional[tuple[int, ...]] = None,
    *,
    generator: Optional[Generator] = None,
) -> nova.Tensor:
    """
    Returns a tensor with random values from a uniform distribution [low, high).

    Args:
        low (float): Lower bound.
        high (float): Upper bound.
        size (Optional[tuple[int, ...]]): Shape of the output tensor.
        generator (Optional[Generator]): Random number generator.

    Returns:
        nova.Tensor: Random tensor from uniform distribution.

    Examples:
        >>> x = nova.uniform(0, 1, size=(2,2))
        >>> print(x)
    """
    gen = generator or _default_generator
    if size is None:
        size = ()
    data = gen._rng.uniform(low, high, size=size)
    return nova.Tensor(data)
