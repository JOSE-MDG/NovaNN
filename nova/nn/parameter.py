"""
This module defines Tensor subclasses used for model parameters and buffers in NovaNN.

Key classes:
- Parameter: Tensor that tracks gradients for learnable weights.
- Buffer: Tensor that does not require gradients but can be saved in the model state.
- UninitializedParameter / UninitializedBuffer: Placeholders for lazy initialization.
- UninitializedTensorMixin: Common base for all uninitialized tensors.

Examples:
    >>> import nova
    >>> from nova.nn import Parameter, Buffer, is_lazy
    >>> p = Parameter(nova.ones((3, 3)))
    >>> b = Buffer(nova.zeros((3,)))
    >>> is_lazy(p)
    False
    >>> lazy_p = UninitializedParameter()
    >>> is_lazy(lazy_p)
    True
    >>> lazy_p_materialized = lazy_p.materialize((3, 3))
    >>> lazy_p_materialized.shape
    (3, 3)
"""

from __future__ import annotations
import nova
from abc import abstractmethod
from typing import Any, TYPE_CHECKING, Optional
from nova import Tensor
from nova.utils import registry_class

if TYPE_CHECKING:
    from nova._typing import Size, Dtype


@registry_class
class Parameter(Tensor):
    """
    A Tensor that is intended to be a learnable model parameter.

    Attributes:
        is_bn_param (bool): Flag indicating if the parameter is part of a batch norm layer.
            Defaults to False.

    Examples:
        >>> import nova
        >>> from nova.nn import Parameter
        >>> p = Parameter(nova.ones((3, 3)))
        >>> p.requires_grad
        True
        >>> p.is_bn_param
        False
    """

    __slots__ = ["is_bn_param"]

    def __init__(
        self, data: Any, requires_grad: bool = True, dtype: Optional[Dtype] = None
    ) -> None:
        super().__init__(
            data=data, requires_grad=requires_grad, dtype=dtype, copy=False
        )
        self.is_bn_param: bool = False


@registry_class
class Buffer(Tensor):
    """
    A Tensor that does not require gradients but can be saved in the model's state.

    Attributes:
        persistent (bool): Whether this buffer is persistent (saved) in the model state.
            Defaults to True.

    Examples:
        >>> import nova
        >>> from nova.nn import Buffer
        >>> b = Buffer(nova.zeros((3,)))
        >>> b.requires_grad
        False
        >>> b.persistent
        True
    """

    __slots__ = ["persistent"]

    def __init__(
        self, data: Any, *, dtype: Optional[Dtype] = None, persistent: bool = True
    ):
        super().__init__(data=data, dtype=dtype, requires_grad=False)
        self.persistent = persistent


@registry_class
class UninitializedTensorMixin:
    """
    Base class for all uninitialized (lazy) tensors.

    Provides a common interface for lazy initialization of parameters and buffers.
    """

    __slots__ = []

    @abstractmethod
    def materialize(self, shape: Size, dtype: Optional[Dtype] = None):
        """Materialize the uninitialized tensor with a specific shape and optional dtype."""
        raise NotImplementedError


@registry_class
class UninitializedParameter(UninitializedTensorMixin, Parameter):
    """
    Lazy/uninitialized parameter that can be materialized later.

    Examples:
        >>> lazy_p = UninitializedParameter()
        >>> lazy_p
        <UninitializedParameter>
        >>> mat = lazy_p.materialize((3, 3))
        >>> mat.shape
        (3, 3)
    """

    __slots__ = []

    def __init__(self, requires_grad: bool = True) -> None:
        dummy_tensor = nova.empty(0)
        Parameter.__init__(self, data=dummy_tensor, requires_grad=requires_grad)

    def materialize(self, size: Size, dtype: Optional[Dtype] = None) -> Parameter:
        tensor = nova.empty(size, dtype=dtype)
        return Parameter(tensor, requires_grad=self.requires_grad)

    def __repr__(self) -> str:
        return "<UninitializedParameter>"


@registry_class
class UninitializedBuffer(UninitializedTensorMixin, Buffer):
    """
    Lazy/uninitialized buffer that can be materialized later.

    Examples:
        >>> lazy_b = UninitializedBuffer()
        >>> lazy_b
        <UninitializedBuffer>
        >>> mat = lazy_b.materialize((2, 2))
        >>> mat.shape
        (2, 2)
    """

    __slots__ = []

    def __init__(self, persistent: bool = True) -> None:
        dummy_tensor = nova.empty(0)
        Buffer.__init__(self, data=dummy_tensor, persistent=persistent)

    def materialize(self, size: Size, dtype: Optional[Dtype] = None) -> Buffer:
        tensor = nova.empty(size, dtype=dtype)
        return Buffer(tensor, persistent=self.persistent)

    def __repr__(self) -> str:
        return "<UninitializedBuffer>"


def is_lazy(param: Any) -> bool:
    """
    Check whether a tensor is an uninitialized (lazy) parameter or buffer.

    Args:
        param (Any): The tensor or object to check.

    Returns:
        bool: True if param is an UninitializedParameter or UninitializedBuffer, False otherwise.

    Examples:
        >>> from nova.nn import Parameter, UninitializedParameter, is_lazy
        >>> p = Parameter(nova.ones((2, 2)))
        >>> is_lazy(p)
        False
        >>> lazy_p = UninitializedParameter()
        >>> is_lazy(lazy_p)
        True
    """
    return isinstance(param, UninitializedTensorMixin)
