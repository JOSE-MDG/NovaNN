import numpy as np
import pandas as pd
from typing import Callable, Iterator, Tuple, Any
from src.core.logger import logger
from src.core.config import (
    EXPORTATION_FASHION_TRAIN_DATA_PATH,
    FASHION_TEST_DATA_PATH,
    FASHION_VALIDATION_DATA_PATH,
    EXPORTATION_MNIST_TRAIN_DATA_PATH,
    MNIST_TEST_DATA_PATH,
    MNIST_VALIDATION_DATA_PATH,
)


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


def load_fashion_mnist_data(
    train_path: str = EXPORTATION_FASHION_TRAIN_DATA_PATH,
    test_path: str = FASHION_TEST_DATA_PATH,
    val_path: str = FASHION_VALIDATION_DATA_PATH,
    normalize: bool = True,
) -> tuple:
    """
    Load Fashion-MNIST dataset from CSV files and optionally normalize it.
    Args:
        train_path: Path to the training data CSV file.
        test_path: Path to the test data CSV file.
        val_path: Path to the validation data CSV file.
        normalize: Whether to normalize the data using training set statistics.
    Returns:
        A tuple containing (x_train, y_train), (x_test, y_test), (x_val, y_val).
    """

    try:
        # Load CSV data using pandas with pyarrow backend for efficiency
        fashion_train = pd.read_csv(train_path, dtype_backend="pyarrow")
        fashion_test = pd.read_csv(test_path, dtype_backend="pyarrow")
        fashion_val = pd.read_csv(val_path, dtype_backend="pyarrow")
        logger.info("Fashion-MNIST data loaded successfully.")

        # Separate features and labels
        x_train = fashion_train.drop(columns=["label"]).values.astype(np.float32)
        y_train = fashion_train["label"].values.astype(np.int64)

        x_test = fashion_test.drop(columns=["label"]).values.astype(np.float32)
        y_test = fashion_test["label"].values.astype(np.int64)

        x_val = fashion_val.drop(columns=["label"]).values.astype(np.float32)
        y_val = fashion_val["label"].values.astype(np.int64)

        # Normalize data if requested
        if normalize:
            # Compute mean and std from training data
            mean = np.mean(x_train, axis=0)
            std = np.std(x_train, axis=0) + 1e-8

            # Apply normalization
            x_train_norm = normalize(x_train, mean, std)
            x_test_norm = normalize(x_test, mean, std)
            x_val_norm = normalize(x_val, mean, std)

            # Return normalized datasets
            return (x_train_norm, y_train), (x_test_norm, y_test), (x_val_norm, y_val)
        else:
            # Return raw datasets
            return (x_train, y_train), (x_test, y_test), (x_val, y_val)

    # Handle exceptions during data loading
    except Exception as e:
        logger.error(f"Error loading Fashion-MNIST data: {e}")


def load_mnist_data(
    train_path: str = EXPORTATION_MNIST_TRAIN_DATA_PATH,
    test_path: str = MNIST_TEST_DATA_PATH,
    val_path: str = MNIST_VALIDATION_DATA_PATH,
    normalize: bool = True,
) -> tuple:
    """
    Load MNIST dataset from CSV files and optionally normalize it.
    Args:
        train_path: Path to the training data CSV file.
        test_path: Path to the test data CSV file.
        val_path: Path to the validation data CSV file.
        normalize: Whether to normalize the data using training set statistics.
    Returns:
        A tuple containing (x_train, y_train), (x_test, y_test), (x_val, y_val).
    """
    try:
        # Load CSV data using pandas with pyarrow backend for efficiency
        mnist_train = pd.read_csv(train_path, dtype_backend="pyarrow")
        mnist_test = pd.read_csv(test_path, dtype_backend="pyarrow")
        mnist_val = pd.read_csv(val_path, dtype_backend="pyarrow")
        logger.info("MNIST data loaded successfully.")

        # Separate features and labels
        x_train = mnist_train.drop(columns=["label"]).values.astype(np.float32)
        y_train = mnist_train["label"].values.astype(np.int64)

        x_test = mnist_test.drop(columns=["label"]).values.astype(np.float32)
        y_test = mnist_test["label"].values.astype(np.int64)

        x_val = mnist_val.drop(columns=["label"]).values.astype(np.float32)
        y_val = mnist_val["label"].values.astype(np.int64)

        # Normalize data if requested
        if normalize:
            # Compute mean and std from training data
            mean = np.mean(x_train, axis=0)
            std = np.std(x_train, axis=0) + 1e-8

            # Apply normalization with training stastistics
            x_train_norm = normalize(x_train, mean, std)
            x_test_norm = normalize(x_test, mean, std)
            x_val_norm = normalize(x_val, mean, std)

            # Return normalized datasets
            return (x_train_norm, y_train), (x_test_norm, y_test), (x_val_norm, y_val)
        else:
            # Return raw datasets
            return (x_train, y_train), (x_test, y_test), (x_val, y_val)

    # Handle exceptions during data loading
    except Exception as e:
        logger.error(f"Error loading MNIST data: {e}")
