from __future__ import annotations
import nova
import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from nova import Tensor


def grad_check_wrt_inputs(
    fn: Callable[[Tensor], Tensor],
    *args: Tensor,
    eps=1e-4,
    zero_grads: bool = True,
    **kwargs,
) -> tuple[list[ndarray], list[ndarray]]:
    """
    Numerical gradient checking for your autograd engine.
    """
    for x in args:
        if isinstance(x, nova.Tensor) and x.requires_grad:
            x.zero_grad()

    # --- Forward ---
    y = fn(*args, **kwargs)

    grad_output = np.ones_like(y.data, dtype=y.dtype)
    y.backward(grad_output)

    # --- Analytical gradients ---
    analytic_grads = []
    for x in args:
        analytic_grads.append(np.copy(x.grad) if x.requires_grad else None)

    # --- Numerical gradients ---
    numerical_grads = []

    for x in args:
        if not x.requires_grad:
            numerical_grads.append(None)
            continue

        grad_num = np.zeros_like(x.data, dtype=x.dtype)

        it = np.nditer(x.data, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            index = it.multi_index
            orig = x.data[index]
            with nova.no_grad():
                x.data[index] = orig + eps
                y_pos = fn(*args, **kwargs).data.copy()

                x.data[index] = orig - eps
                y_neg = fn(*args, **kwargs).data.copy()

                x.data[index] = orig

            grad_num[index] = np.sum((y_pos - y_neg) * grad_output) / (2 * eps)
            it.iternext()

        numerical_grads.append(grad_num)

    if zero_grads:
        for input in args:
            if input.requires_grad:
                input.zero_grad()

    return analytic_grads, numerical_grads
