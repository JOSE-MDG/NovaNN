from __future__ import annotations
import nova
import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from nova import Tensor


def gradient_wrt_x(
    f: Callable[[Tensor], Tensor], x: Tensor, eps: float = 1e-8
) -> ndarray:

    gradient = np.zeros_like(x.data)

    iter = np.nditer(f(x).data, flags=["multi_index"], op_flags=["readwrite"])

    while not iter.finished:

        index = iter.multi_index
        orig = x.data[index]

        with nova.no_grad():
            x.data[index] = orig + eps
            pos = f(x).data.copy()

            x.data[index] = orig - eps
            neg = f(x).data.copy()

            x.data[index] = orig

        gradient[index] = (pos - neg) / 2 * eps

        iter.iternext()

    return gradient
