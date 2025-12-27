from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
import numpy as np


if TYPE_CHECKING:
    from nova.nn import Parameter


def clip_grad_value_(parameters: Iterable[Parameter], clip_value: float):

    params = list(parameters)

    for param in params:
        if param.grad is not None:
            np.clip(param.grad, a_min=-clip_value, a_max=clip_value)
