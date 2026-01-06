from __future__ import annotations
from numpy import ndarray
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nova import Tensor

GradOut = Optional[Tensor | ndarray | list[Tensor | ndarray]]


def grad(
    outputs: Tensor | list[Tensor],
    inputs: Tensor | list[Tensor],
    grad_outputs: GradOut = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> list[ndarray] | ndarray:
    """
    Computes gradients of outputs with respect to inputs.

    Args:
        outputs (Tensor | list[Tensor]): Tensor or list of tensors to differentiate.
        inputs (Tensor | list[Tensor]): Tensor or list of tensors with respect to which to compute gradients.
        grad_outputs (GradOut): Gradients with respect to outputs. If None, assumed to be ones.
        retain_graph (bool): If False, the graph is freed after backward (default PyTorch behavior).
        create_graph (bool): If True, graph of derivatives is constructed (for higher-order derivatives).
        allow_unused (bool): If True, returns None for unused inputs instead of raising error.

    Returns:
        List of gradients (or single gradient if inputs was a single Tensor).

    Examples:
        >>> x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = (x ** 2).sum()
        >>> grads = nova.grad(y, x)
        >>> print(grads)  # [2.0, 4.0, 6.0]
    """
    # Normalize inputs to list
    single_input = not isinstance(inputs, (list, tuple))
    if single_input:
        inputs = [inputs]

    # Normalize outputs to list
    single_output = not isinstance(outputs, (list, tuple))
    if single_output:
        outputs = [outputs]

    # Normalize grad_outputs to list
    if grad_outputs is not None:
        if not isinstance(grad_outputs, (list, tuple)):
            grad_outputs = [grad_outputs]
        if len(grad_outputs) != len(outputs):
            raise ValueError(
                f"grad_outputs must have {len(outputs)} elements, got {len(grad_outputs)}"
            )

    # Save prev gradients
    prev_grads = {inp: inp.grad for inp in inputs}

    # Clean gradients
    for inp in inputs:
        inp.zero_grad(set_to_none=True)

    # Backward pass for each output
    for i, output in enumerate(outputs):
        if not output.requires_grad:
            raise RuntimeError(f"Output {i} does not require gradients")

        gradient = None
        if grad_outputs is not None:
            gradient = grad_outputs[i]
            if isinstance(gradient, Tensor):
                gradient = gradient.data

        output.backward(
            gradient=gradient, retain_graph=retain_graph or (i < len(outputs) - 1)
        )

    # Recolect gradients
    grads = []
    for inp in inputs:
        if inp.grad is None:
            if allow_unused:
                grads.append(None)
            else:
                raise RuntimeError(
                    f"One of the inputs did not contribute to the output. Set allow_unused=True if this is expected."
                )
        else:
            grads.append(inp.grad.copy() if not create_graph else inp.grad)

    # Restore previous gradients if the graph is not desired
    if not create_graph:
        for inp in inputs:
            inp.grad = prev_grads[inp]

    # Return single gradient if it was single input
    return grads[0] if single_input else grads
