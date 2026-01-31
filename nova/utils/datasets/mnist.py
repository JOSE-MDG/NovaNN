from __future__ import annotations
import nova
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from nova.exceptions import LoadError, DatasetCorruptionError
from nova.utils.logger import logger
from nova.utils.data import (
    Dataset,
    normalize,
    split_features_and_labels,
    download_dataset,
    split_validation_dataset,
)

from nova.core import (
    EXPORTATION_MNIST_TRAIN_DATA_PATH,
    MNIST_TRAIN_DATA_PATH,
    MNIST_TEST_DATA_PATH,
    MNIST_VALIDATION_DATA_PATH,
)


if TYPE_CHECKING:
    from nova._typing import Dtype
    from nova import Tensor


class MnistData(Dataset):
    """
    MNIST dataset wrapper.

    Args:
        x: Feature tensor (images)
        y: Label tensor
    """

    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]


def _load_parquet_safe(
    path: Path,
    expected_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Safely load parquet file with validation.

    Args:
        path: Path to parquet file
        expected_samples: Expected number of samples

    Returns:
        Loaded DataFrame

    Raises:
        LoadError: If loading or validation fails
    """
    try:
        df = pd.read_parquet(path, dtype_backend="pyarrow")

        # Basic validation
        if df.empty:
            raise DatasetCorruptionError(f"File {path.name} is empty")

        if expected_samples is not None and len(df) != expected_samples:
            logger.warning(
                f"Expected {expected_samples} samples in {path.name}, got {len(df)}"
            )

        # Check for required columns
        if "label" not in df.columns:
            raise DatasetCorruptionError(f"File {path.name} missing 'label' column")

        # Check we have pixel columns
        pixel_cols = [col for col in df.columns if col.startswith("pixel")]
        if len(pixel_cols) != 784:
            raise DatasetCorruptionError(
                f"File {path.name} has {len(pixel_cols)} pixel columns, expected 784"
            )

        return df

    except pd.errors.ParserError as e:
        raise DatasetCorruptionError(
            f"Failed to parse {path.name}: file may be corrupted"
        ) from e
    except Exception as e:
        raise LoadError(f"Failed to load {path.name}: {e}") from e


def _ensure_dataset_downloaded(
    train_path: Path,
    test_path: Path,
    val_path: Path,
    force_download: bool = False,
    user_provided_paths: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ensure datasets are downloaded and loaded.

    Args:
        train_path: Path to training data
        test_path: Path to test data
        val_path: Path to validation data
        force_download: Force redownload even if files exist
        user_provided_paths: Whether user explicitly provided custom paths

    Returns:
        Tuple of (train_df, test_df, val_df)

    Raises:
        LoadError: If download or loading fails
    """
    paths = [train_path, val_path, test_path]
    all_exist = all([path.exists() for path in paths])

    # NOVA FLOW: If the user provided routes manually and they do not exist, ERROR: No download
    if user_provided_paths and not all_exist:
        missing = [str(p) for p in paths if not p.exists()]
        raise LoadError(
            "User-specified paths do not exist:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nPlease ensure these files exist or remove custom paths to use auto-download."
        )

    if not all_exist or force_download:
        logger.info(
            "MNIST data not found locally. " "Downloading and preparing datasets..."
        )

        try:
            # Div factor for split (10,000 validation / 60,000 total)
            factor = 0.16666666666666667

            # Paths
            data_path = MNIST_TRAIN_DATA_PATH.parent.parent

            # Ensure data directory is writable
            try:
                data_path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise LoadError(
                    f"Cannot create data directory '{data_path}': {e}\n"
                    f"Please check permissions."
                ) from e

            # Download train and test datasets
            download_dataset(
                "mnist",
                root=data_path,
                format="parquet",
                force_redownload=force_download,
            )

            # Load and validate the downloaded training data
            logger.info("Loading downloaded training data...")
            mnist_train_full = _load_parquet_safe(
                MNIST_TRAIN_DATA_PATH,
                expected_samples=60000,
            )

            # Split into train and validation
            logger.info("Creating validation split...")
            mnist_train, mnist_val = split_validation_dataset(
                mnist_train_full,
                label="label",
                factor=factor,
                root=data_path / "Mnist",
                save_method="parquet",
                set_name="mnist_train_e",
                val_name="mnist_validation",
                random_state=8,
                shuffle=True,
                stratify=True,
            )

            # Load test data
            logger.info("Loading test data...")
            mnist_test = _load_parquet_safe(
                MNIST_TEST_DATA_PATH,
                expected_samples=10000,
            )

            logger.info("✓ MNIST data successfully prepared")
            return mnist_train, mnist_test, mnist_val

        except Exception as e:
            logger.error(f"Failed to download/prepare MNIST data: {e}")
            raise LoadError(
                "Failed to download MNIST dataset. "
                "Please check your internet connection and try again."
            ) from e

    else:
        # Load existing files
        logger.info("Loading MNIST data from disk...")

        try:
            mnist_train = _load_parquet_safe(
                EXPORTATION_MNIST_TRAIN_DATA_PATH,
                expected_samples=50000,  # After validation split
            )
            mnist_test = _load_parquet_safe(
                MNIST_TEST_DATA_PATH,
                expected_samples=10000,
            )
            mnist_val = _load_parquet_safe(
                MNIST_VALIDATION_DATA_PATH,
                expected_samples=10000,
            )

            logger.info("✓ MNIST data successfully loaded")
            return mnist_train, mnist_test, mnist_val

        except (DatasetCorruptionError, LoadError) as e:
            logger.warning(f"Existing files appear corrupted: {e}")
            logger.info("Attempting to redownload...")

            # Recursively call with force_download=True
            return _ensure_dataset_downloaded(
                train_path, test_path, val_path, force_download=True
            )


def load_mnist_data(
    tensor4d: bool = False,
    as_tensor: bool = True,
    do_normalize: bool = True,
    dtype: Optional[Dtype] = None,
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
    force_download: bool = False,
) -> tuple[MnistData, MnistData, MnistData]:
    """
    Load MNIST dataset with robust error handling and validation.

    This function:
    - Automatically downloads data if not present
    - Validates data integrity
    - Handles corrupted files by redownloading
    - Provides clear error messages
    - Supports multiple data formats

    Args:
        tensor4d (bool): If True, reshape to (N, 1, 28, 28). Default False.
        as_tensor (bool): If True, convert to NovaNN Tensors. Default True.
        do_normalize (bool): Whether to normalize using training set statistics. Default True.
        dtype (Optional[Dtype]): Data type for input tensors. Default None (uses float32).
        train_path (Path): Path to training file. Default None.
        test_path (Path): Path to test file. Default None.
        val_path (Path): Path to validation file. Default None.
        force_download (bool): Force redownload even if files exist. Default False.

    Returns:
        tuple: (train, test, val) where each is a MnistData instance

    Raises:
        LoadError: If data loading fails after all recovery attempts
        DatasetCorruptionError: If data validation fails

    Examples:
        >>> from nova.utils.datasets import mnist
        >>> train, test, val = mnist.load_mnist_data(
        ...     as_tensor=True,
        ...     tensor4d=True,
        ...     dtype=nova.float32,
        ... )
        >>> train.x.shape
        (50000, 1, 28, 28)
        >>> train.y.dtype
        int64
        >>> test.x.dtype
        float32

        # Force redownload if you suspect corruption
        >>> train, test, val = mnist.load_mnist_data(
        ...     force_download=True
        ... )
    """
    try:
        # Set default dtype
        if dtype is None:
            dtype = nova.float32

        # NOVA FLOW: Detect if the user provided manual routes
        user_provided_paths = any(
            [train_path is not None, test_path is not None, val_path is not None]
        )

        train_path = Path(train_path or MNIST_TRAIN_DATA_PATH)
        test_path = Path(test_path or MNIST_TEST_DATA_PATH)
        val_path = Path(val_path or MNIST_VALIDATION_DATA_PATH)

        if not user_provided_paths:
            for path in [train_path, test_path, val_path]:
                path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure data is downloaded and load it
        mnist_train, mnist_test, mnist_val = _ensure_dataset_downloaded(
            train_path=train_path,
            test_path=test_path,
            val_path=val_path,
            force_download=force_download,
            user_provided_paths=user_provided_paths,
        )

        # Split features and labels
        logger.info("Extracting features and labels...")
        try:
            x_train, y_train = split_features_and_labels(mnist_train)
            x_test, y_test = split_features_and_labels(mnist_test)
            x_val, y_val = split_features_and_labels(mnist_val)
        except Exception as e:
            raise DatasetCorruptionError(
                f"Failed to split features and labels: {e}"
            ) from e

        # Reshape to 4D tensor if requested
        if tensor4d:
            logger.info("Reshaping to 4D tensors (N, 1, 28, 28)...")
            n_train, n_test, n_val = x_train.shape[0], x_test.shape[0], x_val.shape[0]

            x_train = x_train.reshape(n_train, 1, 28, 28)
            x_test = x_test.reshape(n_test, 1, 28, 28)
            x_val = x_val.reshape(n_val, 1, 28, 28)

        # Normalize data if requested
        if do_normalize:
            logger.info("Normalizing data...")
            # Compute mean and std from training data only
            mean = np.mean(x_train, axis=0, keepdims=True)
            std = np.std(x_train, axis=0, keepdims=True)

            # Apply normalization
            x_train = normalize(x_train, mean, std)
            x_test = normalize(x_test, mean, std)
            x_val = normalize(x_val, mean, std)

            # Convert to specified dtype
            if not as_tensor:
                x_train = x_train.astype(dtype)
                x_test = x_test.astype(dtype)
                x_val = x_val.astype(dtype)

        # Convert to tensors if requested
        if as_tensor:
            logger.info("Converting to NovaNN tensors...")
            x_train = nova.tensor(x_train, dtype=dtype)
            y_train = nova.tensor(y_train, dtype=nova.long)

            x_test = nova.tensor(x_test, dtype=dtype)
            y_test = nova.tensor(y_test, dtype=nova.long)

            x_val = nova.tensor(x_val, dtype=dtype)
            y_val = nova.tensor(y_val, dtype=nova.long)

        # Create dataset objects
        train = MnistData(x_train, y_train)
        test = MnistData(x_test, y_test)
        val = MnistData(x_val, y_val)

        logger.info(
            f"✓ Dataset loaded successfully:\n"
            f"  Train: {len(train)} samples\n"
            f"  Test:  {len(test)} samples\n"
            f"  Val:   {len(val)} samples"
        )

        return train, test, val

    except (LoadError, DatasetCorruptionError):
        raise

    except Exception as e:
        logger.error("Unexpected error during data loading:")
        logger.error(traceback.format_exc())
        raise LoadError(
            f"An unexpected error occurred during data loading: {type(e).__name__}: {e}\n"
            f"Please check the logs for details or try force_download=True"
        ) from e


# Convenience function for quick loading with common settings
def load_mnist_default() -> tuple[MnistData, MnistData, MnistData]:
    """
    Load MNIST with recommended default settings.

    Settings:
    - 4D tensors (N, 1, 28, 28)
    - Normalized data
    - Float32 dtype
    - Validation enabled

    Returns:
        tuple: (train, test, val) datasets

    Example:
        >>> train, test, val = load_mnist_default()
        >>> train.x.shape
        (50000, 1, 28, 28)
    """
    return load_mnist_data(
        tensor4d=True,
        as_tensor=True,
        do_normalize=True,
        dtype=nova.float32,
    )
