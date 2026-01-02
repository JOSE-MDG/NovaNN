from __future__ import annotations
from numpy import ndarray
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova import Tensor


def grad(
    outputs: Tensor | list[Tensor],
    inputs: Tensor | list[Tensor],
    gradients: Optional[Tensor | ndarray | list[Tensor | ndarray]] = None,
    retain_grads: bool = True,
) -> list[ndarray] | ndarray:

    if not isinstance(inputs, list):
        inputs = [inputs]

    elif not isinstance(outputs, list):
        outputs = [outputs]

    if gradients is not None and not isinstance(gradients, (list, tuple)):
        gradients = [gradients]

    prev_grads = {input: input.grad for input in inputs}

    for input in inputs:
        input.zero_grad(set_to_none=True)

    for i, output in enumerate(outputs):
        gradient = gradients[i] if gradients else None
        output.backward(gradient=gradient)

    grads = [inp.grad for inp in inputs]

    if retain_grads:
        for input in inputs:
            input.grad = prev_grads[input]

    return grads[0] if len(grads) == 1 else grads
