import numpy as np
import pandas as pd
from novann._typing import TrainTestEvalSets
from novann.utils.log_config import logger
from novann.utils.data import normalize, split_features_and_labels
from novann.core import (
    EXPORTATION_MNIST_TRAIN_DATA_PATH,
    MNIST_TEST_DATA_PATH,
    MNIST_VALIDATION_DATA_PATH,
)


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
        x_train, y_train = split_features_and_labels(mnist_train)
        x_test, y_test = split_features_and_labels(mnist_test)
        x_val, y_val = split_features_and_labels(mnist_val)

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
            x_train_norm = normalize(x_train, mean, std).astype(np.float32)
            x_test_norm = normalize(x_test, mean, std).astype(np.float32)
            x_val_norm = normalize(x_val, mean, std).astype(np.float32)

            # Return normalized datasets
            return (x_train_norm, y_train), (x_test_norm, y_test), (x_val_norm, y_val)
        else:
            # Return raw datasets
            return (x_train, y_train), (x_test, y_test), (x_val, y_val)

    # Handle exceptions during data loading
    except Exception as e:
        logger.error(f"Error loading MNIST data: {e}")
        raise
