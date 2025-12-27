from __future__ import annotations
import numpy as np
import nova
from typing import TYPE_CHECKING, Optional, Type
from numpy import ndarray

if TYPE_CHECKING:
    from nova import Tensor


def _bulid_topo(self: Type[Tensor]):

    topo: list[Tensor] = []
    visited: set[Tensor] = set()
    stack: list[Tensor] = [self]

    while stack:
        node = stack.pop()

        if node not in visited:
            visited.add(node)

            if node.grad_fn is not None:
                for input in node._inputs:
                    if isinstance(input, nova.Tensor):
                        stack.append(input)

            topo.append(node)

    topo.sort(key=lambda x: x.rank)
    return topo


def _backward(
    cls: Type[Tensor],
    gradient: Optional[ndarray | Tensor] = None,
    retain_graph: bool = False,
) -> None:
    cls.grad = gradient

    topo_order = _bulid_topo(cls)

    for tensor in reversed(topo_order):

        if tensor.grad_fn is None:
            continue

        grad_inputs = tensor.grad_fn.backward(tensor._ctx, tensor.grad)

        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)

        for inputs, grad in zip(tensor._inputs, grad_inputs):

            if inputs.requires_grad and grad is not None:

                grad_broadcasted = np.broadcast_to(grad, inputs.data.shape).copy()

                if inputs.grad is None:
                    inputs.grad = grad_broadcasted
                else:
                    inputs.grad += grad_broadcasted

    for tensor in topo_order:

        if tensor.grad is not None and len(tensor._backward_hooks) > 0:
            tensor.grad = tensor._apply_hooks(tensor.grad)

    for tensor in topo_order:

        if not tensor._is_leaf and not tensor.requires_grad:
            tensor.grad = None

        if not retain_graph:
            if not tensor._retain_grad and not tensor._is_leaf:
                tensor.grad_fn = None
                tensor._inputs = []
                tensor._ctx = None
