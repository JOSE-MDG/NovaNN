from __future__ import annotations
import numpy as np
import nova
from typing import TYPE_CHECKING, Optional
from numpy import ndarray

if TYPE_CHECKING:
    from nova import Tensor


def _build_topo(self: Tensor) -> list[Tensor]:
    """
    Builds topological order of tensors in the computational graph.

    Uses iterative depth-first traversal to construct a topological ordering
    of all tensors reachable from the given tensor. This ordering ensures
    that gradients are computed in the correct sequence during backpropagation
    (i.e., a tensor's gradients are computed before its inputs' gradients).

    The algorithm:
    1. Starts from the output tensor (self)
    2. Traverses backward through the computation graph via _inputs
    3. Visits each tensor exactly once
    4. Returns tensors in reverse topological order (output → inputs)

    Args:
        self (Tensor): Output tensor to start traversal from.

    Returns:
        List of tensors in topological order (from output to leaf nodes).

    Notes:
        - Uses iterative approach to avoid Python recursion limit
        - Only includes tensors that are part of the gradient computation
        - Skips tensors without grad_fn (leaf nodes or detached)

    Examples:
        >>> x = nova.tensor([1.0], requires_grad=True)
        >>> y = x * 2
        >>> z = y + 3
        >>> topo = _build_topo(z)
        >>> print([t.grad_fn.__name__ for t in topo])  # ['Add', 'Mul']
    """
    topo: list[Tensor] = []
    visited: set[Tensor] = set()
    stack: list[tuple[Tensor, bool]] = [(self, False)]

    while stack:
        node, processed = stack.pop()

        if processed:
            topo.append(node)
            continue

        if node in visited:
            continue

        visited.add(node)
        stack.append((node, True))

        # Traverse inputs in reverse order for correct gradient accumulation
        if node.grad_fn is not None:
            for input_tensor in reversed(node._inputs):
                if (
                    isinstance(input_tensor, nova.Tensor)
                    and input_tensor not in visited
                ):
                    stack.append((input_tensor, False))

    return topo


def _backward(
    cls: Tensor,
    gradient: Optional[ndarray | Tensor] = None,
    retain_graph: bool = False,
) -> None:
    """
    Executes the backward pass to compute gradients.

    This is the core backpropagation engine that computes gradients of the
    output tensor with respect to all leaf tensors in the computational graph.

    The backward pass proceeds in several phases:

    1. **Initialization**: Sets the output gradient (defaults to ones)
    2. **Topological Sort**: Orders tensors from output to inputs
    3. **Gradient Propagation**: For each tensor in reverse topo order:
       - Calls its grad_fn.backward() to compute input gradients
       - Broadcasts gradients to match input shapes
       - Accumulates gradients for tensors used multiple times
    4. **Hook Application**: Applies registered backward hooks to gradients
    5. **Cleanup**: Removes non-leaf gradients unless retain_graph=True

    Args:
        cls: Output tensor to backpropagate from (typically loss).
        gradient: Initial gradient (dL/d_output). If None, uses ones_like(output).
        retain_graph: If True, preserves graph for multiple backward passes.
            If False, frees memory by clearing grad_fn and _inputs.

    Notes:
        - Gradients accumulate automatically for tensors used multiple times
        - Non-leaf tensors have their gradients cleared unless retain_graph=True
        - Hooks are applied after gradient computation but before cleanup
        - Broadcasting handles shape mismatches between gradient and input

    Examples:
        >>> x = nova.tensor([1.0, 2.0], requires_grad=True)
        >>> y = (x ** 2).sum()
        >>> _backward(y)  # Computes x.grad = [2.0, 4.0]

        >>> # Multiple backward passes
        >>> _backward(y, retain_graph=True)
        >>> x.zero_grad()
        >>> _backward(y)  # Graph still exists
    """
    # Initialize output gradient
    cls.grad = gradient

    # Build topological order of computation graph
    topo_order = _build_topo(cls)

    # Phase 1: Propagate gradients backward through the graph
    for tensor in reversed(topo_order):
        if tensor.grad_fn is None:
            continue

        # Compute gradients with respect to inputs
        grad_inputs = tensor.grad_fn.backward(tensor._ctx, tensor.grad)

        # Ensure grad_inputs is a tuple
        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)

        # Distribute gradients to inputs
        for i, (inputs, grad) in enumerate(zip(tensor._inputs, grad_inputs)):
            if inputs.requires_grad and grad is not None:
                # Broadcast gradient to match input shape (handles reduction ops)
                grad_broadcasted = np.broadcast_to(grad, inputs.data.shape).copy()

                # Accumulate gradients (for tensors used multiple times)
                if inputs.grad is None:
                    inputs.grad = grad_broadcasted
                else:
                    inputs.grad += grad_broadcasted

    # Phase 2: Apply backward hooks to gradients
    for tensor in topo_order:
        if tensor.grad is not None and len(tensor._backward_hooks) > 0:
            tensor.grad = tensor._apply_hooks(tensor.grad)

    # Phase 3: Clean up non-leaf gradients and free graph if not retained
    for tensor in topo_order:
        # Remove gradients from non-leaf intermediate tensors
        if not tensor._is_leaf and not tensor.requires_grad:
            tensor.grad = None

        # Free computational graph unless explicitly retained
        if not retain_graph:
            if not tensor._retain_grad and not tensor._is_leaf:
                tensor.grad_fn = None
                tensor._inputs = []
                tensor._ctx = None
