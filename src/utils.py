# filepath: /home/juancho_col/Documents/Neural Network/src/utils.py
import numpy as np
from typing import Callable, Iterator, Tuple, Any, Optional


def accuracy(
    model: Callable[[np.ndarray], np.ndarray],
    data_loader: Iterator[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """Compute classification accuracy for a model over a dataloader.

    Args:
        model: Callable that maps a batch of inputs X (np.ndarray) to logits/probabilities.
        data_loader: Iterator that yields (X_batch, y_batch) tuples. y_batch should
            contain integer class labels.

    Returns:
        Fraction of correctly predicted samples (float in [0, 1]).
    """
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in data_loader:
        y_pred = model(X_batch)
        pred_classes = np.argmax(y_pred, axis=1)
        total_correct += np.sum(pred_classes == y_batch)
        total_samples += y_batch.shape[0]

    return total_correct / total_samples


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
    grad = np.zeros_like(x, dtype=float)
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
    grad = np.zeros_like(x, dtype=float)
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
    grad = np.zeros_like(x, dtype=float)
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
    grad = np.zeros_like(p.data, dtype=float)
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


def normalize(x_data: np.ndarray, x_mean: np.float32, x_std: np.float32) -> np.ndarray:
    """Normalize input data using provided mean and standard deviation.

    Args:
        x_data: Input data array to normalize.
        x_mean: Mean value for normalization.
        x_std: Standard deviation for normalization.
    Returns:
        Normalized data array.
    """
    return (x_data - x_mean) / x_std
