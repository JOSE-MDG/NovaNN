"""
Gradient clipping utilities for NovaNN.

This module provides functions to clip gradients of parameters by **norm** or **value**.
Useful to prevent exploding gradients during training.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Optional
import numpy as np

if TYPE_CHECKING:
    from nova.nn import Parameter


def clip_grad_norm_(
    parameters: Iterable[Parameter],
    max_norm: Optional[float] = 1.0,
    get_norm: bool = False,
) -> Optional[float]:
    """
    Clip the gradients of parameters by their global L2 norm.

    Args:
        parameters (Iterable[Parameter]): Iterable of parameters to clip.
        max_norm (Optional[float]): Maximum allowed norm for gradients. Defaults to 1.0.
        get_norm (bool): If True, returns the total norm of the gradients before clipping.

    Returns:
        Optional[float]: The total gradient norm if `get_norm=True`, otherwise None.

    Notes:
        - Gradients are scaled in-place if their norm exceeds `max_norm`.
        - If `max_norm` is None, no clipping is performed (returns 1.0).
        - The function uses a small epsilon (1e-10) to avoid division by zero.

    Examples:
        >>> import nova
        >>> import numpy as np
        >>> from nova.nn import Parameter
        >>> p1 = Parameter(nova.tensor([3.0, 4.0]))
        >>> p1.grad = np.array([3.0, 4.0])
        >>> norm = clip_grad_norm_([p1], max_norm=5.0)
    """
    parameters = list(parameters)
    params = [param for param in parameters if param.grad is not None]
    total_norm = 0.0

    for param in params:
        if max_norm is None:
            return 1.0
        p_norm = np.linalg.norm(param.grad, ord=2)
        total_norm += np.sum(p_norm**2)

    total_norm = np.sqrt(total_norm)
    clip_coeff = max_norm / (total_norm + 1e-10)

    if clip_coeff < 1.0:
        for param in params:
            param.grad *= clip_coeff  # scales gradients in-place

    if get_norm:
        return total_norm


def clip_grad_value_(parameters: Iterable[Parameter], clip_value: float):
    """
    Clip the gradients of parameters element-wise by a maximum absolute value.

    Args:
        parameters (Iterable[Parameter]): Iterable of parameters to clip.
        clip_value (float): Maximum allowed absolute value for each gradient element.

    Notes:
        - Gradients are modified in-place using `numpy.clip`.

    Examples:
        >>> import nova
        >>> import numpy as np
        >>> from nova.nn import Parameter
        >>> p1 = Parameter(nova.tensor([1.0, 2.0, -3.0]))
        >>> p1.grad = np.array([1.0, 5.0, -4.0])
        >>> clip_grad_value_([p1], clip_value=3.0)
        >>> p1.grad
        array([ 1.,  5., -4.])
    """
    params = list(parameters)
    for param in params:
        if param.grad is not None:
            np.clip(param.grad, a_min=-clip_value, a_max=clip_value, out=param.grad)
