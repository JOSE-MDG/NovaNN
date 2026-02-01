from __future__ import annotations
import gzip
import tqdm
import time
import nova
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from numpy import ndarray
from nova.utils.logger import get_logger
from nova.core import MNIST_URLS, FASHION_URLS
from typing import TYPE_CHECKING, Optional, Literal, Callable
from nova.exceptions import (
    SaveError,
    FileNotFoundError,
    DatasetValidationError,
    DatasetDownloadError,
)
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from nova import Tensor
    from nova._typing import Dtype

logger = get_logger()

type SubSets = tuple[
    ndarray | Tensor, ndarray | Tensor, ndarray | Tensor, ndarray | Tensor
]


def normalize(
    x_data: ndarray | Tensor, x_mean: float | Tensor, x_std: float | Tensor
) -> ndarray | Tensor:
    """
    Normalize input data using provided mean and standard deviation.

    Args:
        x_data (ndarray | Tensor): Input data to normalize.
        x_mean (float | Tensor): Mean value for normalization.
        x_std (float | Tensor): Standard deviation for normalization.

    Returns:
        ndarray or Tensor: Normalized data.

    Examples:
        >>> import numpy as np
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> normalize(x, x_mean=2.0, x_std=1.0)
        array([-1.,  0.,  1.])
    """
    return (x_data - x_mean) / (x_std + 1e-8)


def split_features_and_labels(
    df: pd.DataFrame, label_column: str = "label", dtype: Optional[Dtype] = None
) -> tuple[ndarray, ndarray]:
    """
    Split a tabular dataset into feature and label arrays.

    Args:
        df (pd.DataFrame): Input dataset.
        label_column (str): Name of the label column. Defaults to "label".
        dtype (Optional[Dtype]): Data type for features. Defaults to np.float32.

    Returns:
        tuple[ndarray, ndarray]: Features array and labels array (int64).

    Notes:
        - If `label_column` does not exist, the first column is assumed to be labels.
        - Features are returned with specified dtype (default float32), labels as int64.

    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'label':[0,1], 'f1':[0.1,0.2], 'f2':[0.3,0.4]})
        >>> x, y = split_features_and_labels(data)
        >>> x
        array([[0.1, 0.3],
               [0.2, 0.4]], dtype=float32)
        >>> y
        array([0, 1], dtype=int64)
    """
    if dtype is None:
        dtype = np.float32

    if label_column in df.columns:
        y = df[label_column].to_numpy(dtype=np.int64)
        x = df.drop(columns=[label_column]).to_numpy(dtype=dtype)
    else:
        y = df.iloc[:, 0].to_numpy(dtype=np.int64)
        x = df.iloc[:, 1:].to_numpy(dtype=dtype)

    return x, y


def split_validation_subset(
    x: ndarray | Tensor,
    y: ndarray | Tensor,
    factor: float,
    shuffle: bool = True,
    stratify: bool = True,
    random_state: Optional[int] = None,
) -> SubSets:
    """
    Split dataset into training and validation subsets.

    Args:
        x (ndarray | Tensor): Feature data.
        y (ndarray | Tensor): Label data.
        factor (float): Fraction of data to use for validation (0.0 to 1.0).
        shuffle (bool): Whether to shuffle data before splitting. Defaults to True.
        stratify (bool): Whether to preserve class distribution. Defaults to True.
        random_state (Optional[int]): Random seed for reproducibility. Defaults to None.

    Returns:
        SubSets: Tuple of (x_train, y_train, x_val, y_val).

    Raises:
        ValueError: If factor is not between 0 and 1.

    Examples:
        >>> import numpy as np
        >>> x = np.random.rand(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> x_train, y_train, x_val, y_val = split_validation_subset(x, y, factor=0.2)
        >>> print(x_train.shape, x_val.shape)
        (80, 10) (20, 10)
    """
    if not 0.0 < factor < 1.0:
        raise ValueError(f"factor must be between 0 and 1, got {factor}")

    tensors = False
    if isinstance(x, nova.Tensor):
        tensors = True
        x = x.detach().numpy()
        y = y.detach().numpy()

    x_set, x_val, y_set, y_val = train_test_split(
        x,
        y,
        test_size=factor,
        shuffle=shuffle,
        stratify=y if stratify else None,
        random_state=random_state,
    )
    if tensors:
        x_set = nova.tensor(x_set, dtype=x_set.dtype)
        y_set = nova.tensor(y_set, dtype=y_set.dtype)
        x_val = nova.tensor(x_val, dtype=x_val.dtype)
        y_val = nova.tensor(y_val, dtype=y_val.dtype)

    return x_set, y_set, x_val, y_val


def _check_root(root: Path) -> None:
    """
    Verify and create directory path if it doesn't exist.

    Args:
        root (Path): Directory path to verify/create.

    Raises:
        FileNotFoundError: If path cannot be created.
    """
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise FileNotFoundError(
            f"Cannot create directory '{root.parent}'. "
            f"Please verify permissions and path validity."
        ) from e


def _validate_dataframe(df: pd.DataFrame, min_rows: int = 1) -> None:
    """
    Validate DataFrame before saving.

    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required

    Raises:
        ValueError: If DataFrame is invalid
    """
    if df is None or df.empty:
        raise ValueError("Cannot save empty DataFrame")

    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum {min_rows} required")

    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        raise ValueError(f"DataFrame has all-null columns: {null_cols}")


def save_to_csv(df: pd.DataFrame, root: Path | str = ".") -> None:
    """
    Save DataFrame to CSV format with validation.

    Args:
        df (pd.DataFrame): DataFrame to save.
        root (Path | str): Output file path. Defaults to current directory.

    Raises:
        SaveError: If saving fails.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> save_to_csv(df, 'output.csv')
    """
    try:
        _validate_dataframe(df)
        root = Path(root)
        _check_root(root)
        df.to_csv(root, index=False)

        # Verify file was created and has content
        if not root.exists() or root.stat().st_size == 0:
            raise SaveError(f"File created but appears empty: {root}")

    except Exception as e:
        # Clean up partial file if it exists
        root = Path(root)
        if root.exists():
            try:
                root.unlink()
            except:  # noqa: E722
                pass
        raise SaveError(f"Failed to save CSV to '{root}': {str(e)}") from e


def save_to_parquet(
    df: pd.DataFrame, root: Path | str = ".", engine: str = "pyarrow"
) -> None:
    """
    Save DataFrame to Parquet format with validation.

    Args:
        df (pd.DataFrame): DataFrame to save.
        root (Path | str): Output file path. Defaults to current directory.
        engine (str): type of backend for pandas. Default pyarrow

    Raises:
        SaveError: If saving fails.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> save_to_parquet(df, 'output.parquet')
    """
    try:
        _validate_dataframe(df)
        root = Path(root)
        _check_root(root)
        df.to_parquet(root, index=False, engine=engine)

        # Verify file was created and has content
        if not root.exists() or root.stat().st_size == 0:
            raise SaveError(f"File created but appears empty: {root}")

    except Exception as e:
        # Clean up partial file if it exists
        root = Path(root)
        if root.exists():
            try:
                root.unlink()
            except:  # noqa: E722
                pass
        raise SaveError(f"Failed to save Parquet to '{root}': {str(e)}") from e


def save_to_excel(
    df: pd.DataFrame, root: Path | str = ".", engine: str = "openpyxl"
) -> None:
    """
    Save DataFrame to Excel format with validation.

    Args:
        df (pd.DataFrame): DataFrame to save.
        root (Path | str): Output file path. Defaults to current directory.
        engine (str): type of backend for pandas. Default openpyxl

    Raises:
        SaveError: If saving fails.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> save_to_excel(df, 'output.xlsx')
    """
    try:
        _validate_dataframe(df)

        # Excel has a row limit
        if len(df) > 1048576:
            raise ValueError(
                f"DataFrame has {len(df)} rows, exceeds Excel limit of 1,048,576"
            )

        root = Path(root)
        _check_root(root)
        df.to_excel(root, index=False, engine=engine)

        # Verify file was created and has content
        if not root.exists() or root.stat().st_size == 0:
            raise SaveError(f"File created but appears empty: {root}")

    except Exception as e:
        # Clean up partial file if it exists
        root = Path(root)
        if root.exists():
            try:
                root.unlink()
            except:  # noqa: E722
                pass
        raise SaveError(f"Failed to save Excel to '{root}': {str(e)}") from e


def split_validation_dataset(
    dataset: pd.DataFrame,
    label: str | list[str],
    factor: float = 0.1,
    *,
    root: Path | str = ".",
    save_method: Literal["csv", "parquet", "excel"] = "csv",
    set_name: str = "train",
    val_name: str = "val",
    shuffle: bool = True,
    stratify: bool = True,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and validation sets and save them.

    Args:
        dataset (pd.DataFrame): Full dataset to split.
        label (str | list[str]): Column name(s) for labels (used for stratification).
        factor (float): Fraction of data for validation (0.0 to 1.0). Default 0.1.
        root (Path | str): Directory where files will be saved. Default current directory.
        save_method (Literal["csv", "parquet", "excel"]): File format. Default "csv".
        set_name (str): Name for training set file (without extension). Default "train".
        val_name (str): Name for validation set file (without extension). Default "val".
        shuffle (bool): Whether to shuffle before splitting. Default True.
        stratify (bool): Whether to preserve label distribution. Default True.
        random_state (Optional[int]): Random seed for reproducibility. Default None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames.

    Raises:
        ValueError: If factor is invalid or dataset is empty.
        SaveError: If saving fails.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'label': [0,0,1,1], 'x': [1,2,3,4]})
        >>> train, val = split_validation_dataset(
        ...     df, label='label', factor=0.25, save_method='parquet'
        ... )
        >>> len(train), len(val)
        (3, 1)
    """

    _validate_dataframe(dataset)

    if not 0.0 < factor < 1.0:
        raise ValueError(f"factor must be between 0 and 1, got {factor}")

    root = Path(root)
    _check_root(root)

    # Validate label column exists
    label_list = [label] if isinstance(label, str) else label
    missing_labels = [l for l in label_list if l not in dataset.columns]  # noqa: E741
    if missing_labels:
        raise ValueError(f"Label columns not found in dataset: {missing_labels}")

    # Perform the split
    stratify_column = dataset[label] if stratify else None

    train, val = train_test_split(
        dataset,
        test_size=factor,
        shuffle=shuffle,
        stratify=stratify_column,
        random_state=random_state,
    )

    # Save files
    save_functions: dict[
        str, tuple[Callable[[pd.DataFrame, Path | str, str], None], str]
    ] = {
        "csv": (save_to_csv, ".csv"),
        "parquet": (save_to_parquet, ".parquet"),
        "excel": (save_to_excel, ".xlsx"),
    }

    if save_method not in save_functions:
        raise ValueError(
            f"Invalid save_method '{save_method}'. Choose from: {list(save_functions.keys())}"
        )

    save_func, ext = save_functions[save_method]

    val_path = root / f"{val_name}{ext}"
    train_path = root / f"{set_name}{ext}"

    save_func(val, val_path)
    save_func(train, train_path)

    return train, val


def _download_with_retry(
    url: str,
    dest: Path,
    max_retries: int = 3,
    timeout: int = 60,
    backoff_factor: float = 2.0,
) -> None:
    """
    Download a file with retry logic and progress indication.

    Args:
        url: URL to download from
        dest: Destination file path
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        backoff_factor: Multiplier for retry delay

    Raises:
        DatasetDownloadError: If all download attempts fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = backoff_factor**attempt
                logger.info(f"  Retry {attempt}/{max_retries} after {delay:.1f}s...")
                time.sleep(delay)

            logger.info(f"  Downloading {dest.name}...")

            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            # Get total size if available
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress
            temp_dest = dest.with_suffix(dest.suffix + ".tmp")

            with open(temp_dest, "wb") as f:
                if total_size:
                    with tqdm.tqdm(
                        desc=f"  {dest.name}",
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))
                else:
                    with tqdm.tqdm(
                        desc=f"  {dest.name}",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))

            # Verify downloaded file is not empty
            if temp_dest.stat().st_size == 0:
                temp_dest.unlink()
                raise DatasetDownloadError("Downloaded file is empty")

            # Move temp file to final destination
            temp_dest.replace(dest)
            logger.info(f"  ✓ Downloaded {dest.name}")
            return

        except (requests.RequestException, IOError, OSError) as e:
            last_exception = e
            logger.warning(f"  Download attempt {attempt + 1} failed: {e}")

            # Clean up partial download
            if dest.exists():
                try:
                    dest.unlink()
                except:  # noqa: E722
                    pass
            temp_dest = dest.with_suffix(dest.suffix + ".tmp")
            if temp_dest.exists():
                try:
                    temp_dest.unlink()
                except:  # noqa: E722
                    pass

    # All retries failed
    raise DatasetDownloadError(
        f"Failed to download {url} after {max_retries} attempts: {last_exception}"
    ) from last_exception


def _validate_idx_file(filepath: Path, expected_magic: int) -> bool:
    """
    Validate IDX file has correct magic number.

    Args:
        filepath: Path to IDX file (may be gzipped)
        expected_magic: Expected magic number

    Returns:
        True if valid, False otherwise
    """
    try:
        with gzip.open(filepath, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            return magic == expected_magic
    except:  # noqa: E722
        return False


def download_dataset(
    dataset: Literal["mnist", "fashion-mnist"],
    root: Path | str,
    format: Literal["csv", "parquet", "xlsx"] = "csv",
    force_redownload: bool = False,
    validate: bool = True,
) -> None:
    """
    Download and convert MNIST or Fashion-MNIST datasets to specified format.

    This function downloads the official dataset files in IDX format from their
    respective servers, converts them to the specified tabular format with proper
    column structure (label + 784 pixel values), and saves them in the appropriate
    directory.

    Directory structure created:
        root/
        ├── Mnist/
        │   ├── mnist_train.{format}
        │   └── mnist_test.{format}
        └── FashionMnist/
            ├── fashion-mnist_train.{format}
            └── fashion-mnist_test.{format}

    Args:
        dataset (Literal["mnist", "fashion-mnist"]): Dataset to download.
            - "mnist": Original MNIST handwritten digits (0-9)
            - "fashion-mnist": Fashion-MNIST clothing items (10 classes)
        root (Path | str): Root directory where datasets will be saved.
        format (Literal["csv", "parquet", "xlsx"]): Output file format.
            - "csv": Comma-separated values (default, most compatible)
            - "parquet": Apache Parquet (fast, compressed, recommended for large datasets)
            - "xlsx": Excel format (convenient for manual inspection)
            Defaults to "csv".
        force_redownload (bool): If True, redownload even if files exist.
            Defaults to False.
        validate (bool): If True, validate IDX files after download.
            Defaults to True.

    Raises:
        DatasetDownloadError: If download from server fails.
        DatasetValidationError: If downloaded files are corrupted.
        IOError: If file operations fail.
        ValueError: If unsupported format is specified.

    Notes:
        - If files already exist in the specified format, they are not re-downloaded
        - Temporary .gz files are automatically cleaned up after conversion
        - Each image is flattened to 784 pixels (28x28)
        - Data format: first column is 'label', followed by 'pixel0' to 'pixel783'
        - Parquet format offers ~10x smaller file sizes compared to CSV
        - Excel format has a limit of 1,048,576 rows (sufficient for these datasets)
        - Failed downloads are retried automatically with exponential backoff
        - Partial downloads are cleaned up on failure

    Examples:
        >>> from nova.utils.data import download_dataset

        # Download as CSV (default)
        >>> download_dataset("mnist")
        Processing mnist train...
          Downloading train-images.gz... ✓
          Downloading train-labels.gz... ✓
        ✓ Saved mnist_train.csv

        # Download as Parquet (recommended for better performance)
        >>> download_dataset("fashion-mnist", format="parquet")
        Processing fashion-mnist train...
        ✓ Saved fashion-mnist_train.parquet

        # Force redownload
        >>> download_dataset("mnist", force_redownload=True)
        Processing mnist train...
        ✓ Saved mnist_train.csv

        # Files are cached - second call with same format is instant
        >>> download_dataset("mnist", format="csv")
        ✓ mnist_train.csv already exists
        ✓ mnist_test.csv already exists
    """
    root = Path(root)
    _check_root(root)

    # Validate format
    valid_formats = {"csv", "parquet", "xlsx"}
    if format not in valid_formats:
        raise ValueError(f"Unsupported format '{format}'. Choose from: {valid_formats}")

    # Configure folder and prefix based on dataset
    if dataset == "mnist":
        folder = root / "Mnist"
        prefix = "mnist"
        urls = MNIST_URLS
    elif dataset == "fashion-mnist":
        folder = root / "FashionMnist"
        prefix = "fashion-mnist"
        urls = FASHION_URLS
    else:
        raise ValueError(f"The dataset '{dataset}' is not yet supported")

    # Verify we can create the folder
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise DatasetDownloadError(
            f"Cannot create dataset directory '{folder}': {e}\n"
            f"Please check permissions."
        ) from e

    # File extension mapping
    extensions = {"csv": ".csv", "parquet": ".parquet", "xlsx": ".xlsx"}
    ext = extensions[format]

    # Magic numbers for validation
    IMAGE_MAGIC = 2051  # For 3D image arrays
    LABEL_MAGIC = 2049  # For 1D label arrays

    # Process both train and test splits
    for split in ["train", "test"]:
        output_path = folder / f"{prefix}_{split}{ext}"

        # Skip if already exists (unless force redownload)
        if output_path.exists() and not force_redownload:
            logger.info(f"✓ {output_path.name} already exists")
            continue

        logger.info(f"Processing {dataset} {split}...")

        # Download IDX files
        images_key = f"{split}-images"
        labels_key = f"{split}-labels"

        images_gz = folder / f"{images_key}.gz"
        labels_gz = folder / f"{labels_key}.gz"

        try:
            # Download with retry logic
            _download_with_retry(urls[images_key], images_gz)
            _download_with_retry(urls[labels_key], labels_gz)

            # Validate downloaded files
            if validate:
                logger.info("  Validating downloaded files...")
                if not _validate_idx_file(images_gz, IMAGE_MAGIC):
                    raise DatasetValidationError(
                        f"Downloaded image file {images_gz.name} is corrupted or invalid"
                    )
                if not _validate_idx_file(labels_gz, LABEL_MAGIC):
                    raise DatasetValidationError(
                        f"Downloaded label file {labels_gz.name} is corrupted or invalid"
                    )
                logger.info("  ✓ Files validated")

            # Read and convert to arrays
            images = _read_images(images_gz)
            labels = _read_labels(labels_gz)

            # Validate data consistency
            if len(images) != len(labels):
                raise DatasetValidationError(
                    f"Mismatch: {len(images)} images but {len(labels)} labels"
                )

            expected_samples = 60000 if split == "train" else 10000
            if len(images) != expected_samples:
                logger.warning(
                    f"Expected {expected_samples} samples but got {len(images)}"
                )

            # Flatten images and create DataFrame
            logger.info("  Converting to DataFrame...")
            images_flat = images.reshape(images.shape[0], -1)
            data = np.column_stack([labels, images_flat])
            columns = ["label"] + [f"pixel{i}" for i in range(784)]

            df = pd.DataFrame(data, columns=columns)

            # Save in the specified format
            logger.info(f"  Saving to {format}...")
            _save_dataframe(df, output_path, format)

            logger.info(f"✓ Saved {output_path.name}\n")

        except Exception:
            # Clean up on failure
            for path in [images_gz, labels_gz, output_path]:
                if path.exists():
                    try:
                        path.unlink()
                        logger.info(f"  Cleaned up {path.name}")
                    except:  # noqa: E722
                        pass
            raise

        finally:
            # Cleanup temporary .gz files on success
            for gz_file in [images_gz, labels_gz]:
                if gz_file.exists():
                    try:
                        gz_file.unlink()
                    except:  # noqa: E722
                        logger.warning(
                            f"  Could not remove temporary file {gz_file.name}"
                        )


def _save_dataframe(
    df: pd.DataFrame, path: Path, format: Literal["csv", "parquet", "xlsx"]
) -> None:
    """
    Save DataFrame in the specified format.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (Path): Output file path.
        format (Literal["csv", "parquet", "xlsx"]): Output format.

    Raises:
        ValueError: If format is not supported.
        IOError: If saving fails.
    """
    if format == "csv":
        save_to_csv(df, path)
    elif format == "parquet":
        save_to_parquet(df, path)
    elif format == "xlsx":
        save_to_excel(df, path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _read_images(filepath: Path) -> np.ndarray:
    """
    Read IDX format image file.

    The IDX format stores images in a binary format with the following header:
    - Magic number (4 bytes): should be 2051 for images
    - Number of images (4 bytes)
    - Number of rows (4 bytes)
    - Number of columns (4 bytes)
    - Followed by pixel data (uint8)

    Args:
        filepath (Path): Path to .gz compressed IDX file.

    Returns:
        np.ndarray: Array of shape (n_images, height, width) with pixel values.

    Raises:
        ValueError: If file format is invalid.

    Examples:
        >>> images = _read_images(Path("train-images-idx3-ubyte.gz"))
        >>> images.shape
        (60000, 28, 28)
        >>> images.dtype
        dtype('uint8')
    """
    try:
        with gzip.open(filepath, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2051:  # Magic number for image files
                raise ValueError(f"Invalid magic number for images: {magic}")

            n = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")

            if rows != 28 or cols != 28:
                raise ValueError(f"Expected 28x28 images, got {rows}x{cols}")

            data = np.frombuffer(f.read(), dtype=np.uint8)
            expected_size = n * rows * cols

            if len(data) != expected_size:
                raise ValueError(f"Expected {expected_size} bytes, got {len(data)}")

            return data.reshape(n, rows, cols)
    except gzip.BadGzipFile as e:
        raise ValueError(f"File {filepath} is not a valid gzip file") from e


def _read_labels(filepath: Path) -> np.ndarray:
    """
    Read IDX format label file.

    The IDX format stores labels in a binary format with the following header:
    - Magic number (4 bytes): should be 2049 for labels
    - Number of labels (4 bytes)
    - Followed by label data (uint8)

    Args:
        filepath (Path): Path to .gz compressed IDX file.

    Returns:
        np.ndarray: Array of shape (n_labels,) with label values.

    Raises:
        ValueError: If file format is invalid.

    Examples:
        >>> labels = _read_labels(Path("train-labels-idx1-ubyte.gz"))
        >>> labels.shape
        (60000,)
        >>> labels.dtype
        dtype('uint8')
        >>> labels[:5]
        array([5, 0, 4, 1, 9], dtype=uint8)
    """
    try:
        with gzip.open(filepath, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2049:  # Magic number for label files
                raise ValueError(f"Invalid magic number for labels: {magic}")

            n = int.from_bytes(f.read(4), "big")
            data = np.frombuffer(f.read(), dtype=np.uint8)

            if len(data) != n:
                raise ValueError(f"Expected {n} labels, got {len(data)}")

            return data
    except gzip.BadGzipFile as e:
        raise ValueError(f"File {filepath} is not a valid gzip file") from e
