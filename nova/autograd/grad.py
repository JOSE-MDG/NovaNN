from __future__ import annotations
from numpy import ndarray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor


def grad(
    outputs: Tensor | list[Tensor],
    inputs: Tensor | list[Tensor],
    gradients: Tensor | ndarray | list[Tensor | ndarray],
    retain_grads: bool = True,
) -> list[ndarray] | ndarray:

    if None in inputs or None in outputs:
        raise ValueError("The gradient of a None type cannot be calculated")

    if not isinstance(inputs, list):
        inputs = [inputs]

    elif not isinstance(outputs, list):
        outputs = [outputs]

    if gradients is not None and not isinstance(gradients, (list, tuple)):
        gradients = [gradients]

    tensor_ids = {}  # to avoid the loss by memory reference
    id = 0

    for input in inputs:
        if input.grad is not None:
            tensor_ids[input] = id
            id += 1

    prev_grads = {tensor_ids[input]: input.grad for input in inputs}

    for input in inputs:
        input.zero_grad(set_to_none=True)

    for i, output in enumerate(outputs):
        gradient = gradients[i] if gradients else None
        output.backward(gradient=gradient)

    grads = [inp.grad for inp in inputs]

    if retain_grads:
        for i, input in enumerate(inputs):
            input.grad = prev_grads[i]

    return grads[0] if len(grads) == 1 else grads
