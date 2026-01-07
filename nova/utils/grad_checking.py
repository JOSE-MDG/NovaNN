"""
Utilities for verifying gradients of functions in NovaNN.

This module provides functions to perform **numerical gradient checking**,
comparing analytical gradients computed by backpropagation with finite-difference
approximations. It is useful for testing custom operations and ensuring that
gradients are correctly implemented.
"""

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
    eps: float = 1e-4,
    zero_grads: bool = True,
    **kwargs,
) -> tuple[list[ndarray], list[ndarray]]:
    """
    Perform gradient checking of a function with respect to its input tensors.

    This function compares the **analytical gradients** computed by NovaNN's
    automatic differentiation with **numerical gradients** computed via
    finite differences. It is useful for testing custom operations, layers,
    or functions.

    Args:
        fn (Callable[[Tensor], Tensor]): Function whose gradients will be checked.
            It must accept Tensors as inputs and return a single Tensor.
        *args (Tensor): Input tensors to `fn` with `requires_grad=True` if
            gradients are to be checked.
        eps (float, optional): Small perturbation for finite difference approximation.
            Defaults to 1e-4.
        zero_grads (bool, optional): If True, resets gradients of input tensors
            after checking. Defaults to True.
        **kwargs: Additional keyword arguments to pass to `fn`.

    Returns:
        tuple[list[ndarray], list[ndarray]]:
            A tuple of two lists:
            - `analytic_grads`: gradients computed via backpropagation (numpy arrays)
            - `numerical_grads`: gradients computed using finite differences

    Notes:
        - Inputs that do not require gradients will have `None` in the
          corresponding positions of the returned lists.
        - Gradient checking can be slow for large tensors due to the element-wise
          finite difference computation.
        - The function resets gradients of input tensors to zero before and after
          checking to avoid accumulation issues.

    Example:
        >>> import nova
        >>> from nova.utils import grad_check_wrt_inputs
        >>> x = nova.tensor([1.0, 2.0, 3.0], dtype=nova.float32, requires_grad=True)
        >>> def fn(t):
        ...     return (t**2).sum()
        >>> analytic, numeric = grad_check_wrt_inputs(fn, x)
        >>> nova.allclose(analytic[0], numeric[0], rtol=1e-3, atol=1e-5)
        True
    """
    # Zero gradients before starting
    for x in args:
        if isinstance(x, nova.Tensor) and x.requires_grad:
            x.zero_grad(set_to_none=True)

    # Forward pass
    y = fn(*args, **kwargs)

    # Check output is scalar or can be reduced
    if y.numel() == 1:
        grad_output = np.array([1.0], dtype=y.dtype)
    else:
        # For non-scalar outputs, use ones as grad_output
        grad_output = np.ones_like(y.data, dtype=y.dtype)

    # Backward pass
    y.backward(grad_output, retain_graph=False)

    # Store analytical gradients (copy them before they get cleared)
    analytic_grads = []
    for x in args:
        if x.requires_grad:
            if x.grad is not None:
                analytic_grads.append(np.copy(x.grad))
            else:
                # If grad is None, it means this input doesn't contribute
                analytic_grads.append(np.zeros_like(x.data))
        else:
            analytic_grads.append(None)

    numerical_grads = []
    for idx, x in enumerate(args):
        if not x.requires_grad:
            numerical_grads.append(None)
            continue

        # Initialize numerical gradient array
        grad_num = np.zeros_like(x.data, dtype=x.dtype)

        # Iterate over all indices of the tensor
        it = np.nditer(x.data, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            index = it.multi_index
            orig = x.data[index].copy()  # Save original value

            # Perturb +eps
            x.data[index] = orig + eps
            with nova.no_grad():
                y_pos = fn(*args, **kwargs).data.copy()

            # Perturb -eps
            x.data[index] = orig - eps
            with nova.no_grad():
                y_neg = fn(*args, **kwargs).data.copy()

            # Restore original value
            x.data[index] = orig

            # Central difference formula
            # grad = (f(x+eps) - f(x-eps)) / (2*eps)
            grad_num[index] = np.sum((y_pos - y_neg) * grad_output) / (2 * eps)
            it.iternext()

        numerical_grads.append(grad_num)

    # Optionally zero gradients after checking
    if zero_grads:
        for x in args:
            if x.requires_grad:
                x.zero_grad(set_to_none=True)

    return analytic_grads, numerical_grads
