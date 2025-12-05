import numpy as np
from typing import Callable, Any


def numeric_grad_elementwise(
    act_forward: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Compute elementwise numerical gradient of a vector-valued function.

    This computes d act_forward(x) / d x elementwise using central differences.
    The input array `x` is temporarily modified but restored to its original values.

    Args:
        act_forward: Callable that accepts `x` and returns an array of the same shape.
        x: Input array to differentiate with respect to.
        eps: Small perturbation used for finite differences.

    Returns:
        Array with the same shape as `x` containing the numerical gradient.
    """
    grad = np.zeros_like(x, dtype=np.float32)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = act_forward(x).copy()
        x[idx] = orig - eps
        f_minus = act_forward(x).copy()
        grad[idx] = (f_plus[idx] - f_minus[idx]) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def numeric_grad_scalar_from_softmax(
    softmax_forward: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    G: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Numerical gradient of scalar loss L = sum(softmax(x) * G) with respect to x.

    Useful for testing softmax + cross-entropy style Jacobian-vector products.

    Args:
        softmax_forward: Callable that returns softmax probabilities for input x.
        x: Input logits array to differentiate with respect to.
        G: Gradient / weighting matrix used to form the scalar L = sum(softmax(x) * G).
        eps: Finite difference step.

    Returns:
        Numerical gradient dL/dx with same shape as `x`.
    """
    grad = np.zeros_like(x, dtype=np.float32)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        L_plus = np.sum(softmax_forward(x) * G)
        x[idx] = orig - eps
        L_minus = np.sum(softmax_forward(x) * G)
        grad[idx] = (L_plus - L_minus) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def numeric_grad_scalar_wrt_x(
    forward_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    G: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Numerical gradient of scalar S = sum(forward_fn(x) * G) w.r.t. x.

    Generic helper for computing dS/dx using central differences.

    Args:
        forward_fn: Callable producing an array of same shape as x.
        x: Input array.
        G: Weighting array used to form scalar S = sum(forward_fn(x) * G).
        eps: Finite difference epsilon.

    Returns:
        Numerical gradient array with the same shape as `x`.
    """
    grad = np.zeros_like(x, dtype=np.float32)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        L_plus = np.sum(forward_fn(x) * G)
        x[idx] = orig - eps
        L_minus = np.sum(forward_fn(x) * G)
        grad[idx] = (L_plus - L_minus) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def numeric_grad_wrt_param(
    layer: Any, param_attr: str, x: np.ndarray, G: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Numerical gradient of scalar S = sum(layer.forward(x) * G) w.r.t. a layer parameter.

    Args:
        layer: Layer object that exposes a Parameters-like attribute named `param_attr`
               (i.e., has `.data` which is a numpy array).
        param_attr: Attribute name on `layer` that returns the Parameters wrapper.
        x: Input array passed to `layer.forward`.
        G: Weighting array used to form the scalar S = sum(output * G).
        eps: Finite difference step.

    Returns:
        Numerical gradient array with same shape as the parameter `.data`.
    """
    p = getattr(layer, param_attr)
    grad = np.zeros_like(p.data, dtype=np.float32)
    it = np.nditer(p.data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = p.data[idx]
        p.data[idx] = orig + eps
        L_plus = np.sum(layer.forward(x) * G)
        p.data[idx] = orig - eps
        L_minus = np.sum(layer.forward(x) * G)
        grad[idx] = (L_plus - L_minus) / (2 * eps)
        p.data[idx] = orig
        it.iternext()
    return grad
