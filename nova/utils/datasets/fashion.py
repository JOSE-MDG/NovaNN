from __future__ import annotations
import nova
import traceback
import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING
from nova.utils.log_config import logger
from nova.utils.data import normalize, split_features_and_labels
from nova.core import (
    EXPORTATION_FASHION_TRAIN_DATA_PATH,
    FASHION_TEST_DATA_PATH,
    FASHION_VALIDATION_DATA_PATH,
)


if TYPE_CHECKING:
    from nova._typing import Dtype


def load_fashion_mnist_data(
    do_normalize: bool = True,
    tensor4d: bool = False,
    as_tensor: bool = True,
    dtype: Optional[Dtype] = None,
    train_path: str = EXPORTATION_FASHION_TRAIN_DATA_PATH,
    test_path: str = FASHION_TEST_DATA_PATH,
    val_path: str = FASHION_VALIDATION_DATA_PATH,
):
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
        x_train, y_train = split_features_and_labels(fashion_train)
        x_test, y_test = split_features_and_labels(fashion_test)
        x_val, y_val = split_features_and_labels(fashion_val)

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
            x_train = normalize(x_train, mean, std).astype(dtype)
            x_test = normalize(x_test, mean, std).astype(dtype)
            x_val = normalize(x_val, mean, std).astype(dtype)

        if as_tensor:
            x_train = nova.tensor(x_train, dtype=dtype)
            y_train = nova.tensor(y_train, dtype=nova.long)

            x_test = nova.tensor(x_test, dtype=dtype)
            y_test = nova.tensor(y_test, dtype=nova.long)

            x_val = nova.tensor(x_val, dtype=dtype)
            y_val = nova.tensor(y_val, dtype=nova.long)

        return ((x_train, y_train), (x_test, y_test), (x_val, y_val))

    # Handle exceptions during data loading
    except Exception as e:
        lines = [line for line in traceback.format_exception(e)]
        logger.error(f"Error loading MNIST data \n")
        print(*lines)
