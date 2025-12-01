import numpy as np
import pandas as pd
from typing import Callable, Any
from novann._typing import TrainTestEvalSets, Optimizer, LossFunc
from novann.core import DataLoader
from novann.model import Sequential
from novann.core import logger
from novann.core import (
    EXPORTATION_FASHION_TRAIN_DATA_PATH,
    FASHION_TEST_DATA_PATH,
    FASHION_VALIDATION_DATA_PATH,
    EXPORTATION_MNIST_TRAIN_DATA_PATH,
    MNIST_TEST_DATA_PATH,
    MNIST_VALIDATION_DATA_PATH,
)


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


def _split_features_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split a tabular dataset into features and integer labels.

    This helper is robust to whether the CSV includes an explicit "label"
    column header or not. If a "label" column exists, it is used; otherwise,
    the first column is treated as the label column.
    """
    if "label" in df.columns:
        y = df["label"].to_numpy(dtype=np.int32)
        x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    else:
        # Assume the first column holds labels when no explicit header is present
        y = df.iloc[:, 0].to_numpy(dtype=np.int32)
        x = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return x, y


def load_fashion_mnist_data(
    train_path: str = EXPORTATION_FASHION_TRAIN_DATA_PATH,
    test_path: str = FASHION_TEST_DATA_PATH,
    val_path: str = FASHION_VALIDATION_DATA_PATH,
    do_normalize: bool = True,
    tensor4d: bool = False,
) -> TrainTestEvalSets:
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
        logger.debug("Fashion-MNIST data loaded successfully.")

        # Separate features and labels (support both headered and headerless CSVs)
        x_train, y_train = _split_features_and_labels(fashion_train)
        x_test, y_test = _split_features_and_labels(fashion_test)
        x_val, y_val = _split_features_and_labels(fashion_val)

        if tensor4d:
            # Get the number of samples
            n_train = x_train.shape[0]
            n_test = x_test.shape[0]
            n_val = x_val.shape[0]

            # compose to 4d tensor
            x_train = x_train.reshape(n_train, 1, 28, 28)
            x_test = x_test.reshape(n_test, 1, 28, 28)
            x_val = x_val.reshape(n_val, 1, 28, 28)

        # Normalize data if requested
        if do_normalize:
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
        raise


def load_mnist_data(
    train_path: str = EXPORTATION_MNIST_TRAIN_DATA_PATH,
    test_path: str = MNIST_TEST_DATA_PATH,
    val_path: str = MNIST_VALIDATION_DATA_PATH,
    do_normalize: bool = True,
    tensor4d: bool = False,
) -> TrainTestEvalSets:
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
        logger.debug("MNIST data loaded successfully.")

        # Separate features and labels (support both headered and headerless CSVs)
        x_train, y_train = _split_features_and_labels(mnist_train)
        x_test, y_test = _split_features_and_labels(mnist_test)
        x_val, y_val = _split_features_and_labels(mnist_val)

        if tensor4d:
            # Get the number of samples
            n_train = x_train.shape[0]
            n_test = x_test.shape[0]
            n_val = x_val.shape[0]

            # compose to 4d tensor
            x_train = x_train.reshape(n_train, 1, 28, 28)
            x_test = x_test.reshape(n_test, 1, 28, 28)
            x_val = x_val.reshape(n_val, 1, 28, 28)

        # Normalize data if requested
        if do_normalize:
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
        raise


def train(
    train_loader: DataLoader,
    eval_loader: DataLoader,
    net: Sequential,
    optimizer: Optimizer,
    loss_fn: LossFunc,
    epochs: int,
    show_logs_every: int = 0,
    metric: Callable[[Sequential, DataLoader]] = None,
    verbose: bool = True,
    get_model: bool = True,
):
    # Training mode
    net.train()

    # Training loop
    for epoch in range(epochs):
        for input, target in train_loader:
            # Set gradients to None
            optimizer.zero_grad()

            # Foward pass
            outputs = net(input)

            # Compute loss and gradients
            cost, grad = loss_fn(outputs, target)

            # Backward pass
            net.backward(grad)

            # Update parameters
            optimizer.step()

        # Validation result after each epoch
        if metric is not None:
            net.eval()
            result = metric(net, eval_loader)

        if verbose:
            if show_logs_every > 0:
                if (epoch + 1) % show_logs_every == 0:
                    net.train()
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs}, Loss: {cost:.4f}, Validation {metric.__name__}: {result:.4f}"
                    )
            else:
                net.train()
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {cost:.4f}, Validation {metric.__name__}: {result:.4f}"
                )
    if get_model:
        return net
