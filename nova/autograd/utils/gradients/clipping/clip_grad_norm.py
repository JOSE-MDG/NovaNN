from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Optional
import numpy as np


if TYPE_CHECKING:
    from nova.nn import Parameter


def clip_grad_norm_(
    parameters: Iterable[Parameter],
    max_norm: Optional[float] = 1.0,
    get_norm: bool = False,
):

    parameters = list(parameters)

    params = [param for param in parameters if param.gras is not None]

    total_norm = 0

    for param in params:

        if max_norm is None:
            return 1.0

        p_norm = np.linalg.norm(param, ord=2)
        total_norm += np.sum(p_norm**2)

    total_norm = np.sqrt(total_norm)

    clip_coeff = max_norm / (total_norm + 1e-10)

    if clip_coeff < 1.0:
        for param in params:
            param.grad * clip_coeff

    if get_norm:
        return total_norm
