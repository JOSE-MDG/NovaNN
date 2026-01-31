import pytest
import gzip
import struct
import numpy as np
import pandas as pd
import nova
import requests
from pathlib import Path
from unittest.mock import patch, MagicMock
from nova.utils.data.preprocessing import (
    normalize,
    split_features_and_labels,
    split_validation_subset,
    split_validation_dataset,
    save_to_csv,
    save_to_parquet,
    save_to_excel,
    download_dataset,
    _validate_dataframe,
    _check_root,
    _validate_idx_file,
    _read_images,
    _read_labels,
    _save_dataframe,
    _download_with_retry,
)
from nova.exceptions import (
    SaveError,
    DatasetDownloadError,
)

nova.manual_seed(8)


# Fixtures


@pytest.fixture
def sample_dataframe():
    """Basic DataFrame with label + pixel-like columns."""
    n_samples = 100
    n_features = 784
    data = {
        "label": np.random.randint(0, 10, n_samples),
    }
    for i in range(n_features):
        data[f"pixel{i}"] = np.random.randint(0, 256, n_samples)
    return pd.DataFrame(data)


@pytest.fixture
def small_dataframe():
    """Small DataFrame for quick save/load tests."""
    return pd.DataFrame(
        {
            "label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "f2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )


@pytest.fixture
def tmp_output(tmp_path):
    """Provides a clean temporary directory for file outputs."""
    return tmp_path


def _make_idx_images_gz(path: Path, n: int = 10, rows: int = 28, cols: int = 28):
    """Helper: write a valid IDX image .gz file."""
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, rows, cols))
        f.write(np.random.randint(0, 256, n * rows * cols, dtype=np.uint8).tobytes())


def _make_idx_labels_gz(path: Path, n: int = 10):
    """Helper: write a valid IDX label .gz file."""
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, n))
        f.write(np.random.randint(0, 10, n, dtype=np.uint8).tobytes())


# normalize


class TestNormalize:
    """Test the normalize function."""

    def test_basic_normalization_numpy(self):
        """Test standard normalization with numpy arrays."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = np.mean(x)
        std = np.std(x)

        result = normalize(x, mean, std)

        assert abs(result.mean()) < 1e-6
        assert abs(result.std() - 1.0) < 0.05

    def test_basic_normalization_tensor(self):
        """Test normalization with Nova tensors."""
        x = nova.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = x.mean()
        std = x.std()

        result = normalize(x, mean, std)

        assert abs(result.mean().item()) < 1e-5
        assert abs(result.std().item() - 1.0) < 0.1

    def test_known_values(self):
        """Test with known input/output values."""
        x = np.array([1.0, 2.0, 3.0])
        result = normalize(x, x_mean=2.0, x_std=1.0)

        expected = np.array([-1.0, 0.0, 1.0]) / (1.0 + 1e-8)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_zero_std_no_crash(self):
        """Test that zero std does not cause division by zero (epsilon guard)."""
        x = np.array([5.0, 5.0, 5.0])
        result = normalize(x, x_mean=5.0, x_std=0.0)

        # With epsilon, result should be ~0, not inf/nan
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_2d_array(self):
        """Test normalization on a 2D array (batch of samples)."""
        x = np.random.rand(50, 784).astype(np.float32)
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)

        result = normalize(x, mean, std)

        # Per-feature mean should be ~0
        assert np.allclose(result.mean(axis=0), 0.0, atol=1e-5)

    def test_scalar_mean_and_std(self):
        """Test with scalar float mean and std."""
        x = np.array([10.0, 20.0, 30.0])
        result = normalize(x, x_mean=20.0, x_std=10.0)

        expected = (x - 20.0) / (10.0 + 1e-8)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# split_features_and_labels


class TestSplitFeaturesAndLabels:
    """Test the split_features_and_labels function."""

    def test_basic_split(self, small_dataframe):
        """Test standard split with a 'label' column."""
        x, y = split_features_and_labels(small_dataframe)

        assert x.shape == (10, 2)
        assert y.shape == (10,)
        assert y.dtype == np.int64
        assert x.dtype == np.float32

    def test_label_column_excluded_from_features(self, small_dataframe):
        """Test that the label column is not present in features."""
        x, y = split_features_and_labels(small_dataframe)

        # y should match the 'label' column values
        np.testing.assert_array_equal(y, small_dataframe["label"].to_numpy())

    def test_custom_label_column(self):
        """Test split with a custom label column name."""
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1],
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [5.0, 6.0, 7.0, 8.0],
            }
        )
        x, y = split_features_and_labels(df, label_column="target")

        assert x.shape == (4, 2)
        np.testing.assert_array_equal(y, [0, 1, 0, 1])

    def test_fallback_first_column_when_label_missing(self):
        """Test that when label_column is missing, first column is used as labels."""
        df = pd.DataFrame(
            {"class": [0, 1, 2], "feat1": [1.0, 2.0, 3.0], "feat2": [4.0, 5.0, 6.0]}
        )
        x, y = split_features_and_labels(
            df, label_column="label"
        )  # 'label' doesn't exist

        # Should fall back to first column ('class')
        np.testing.assert_array_equal(y, [0, 1, 2])
        assert x.shape == (3, 2)

    def test_custom_dtype(self, small_dataframe):
        """Test that custom dtype is applied to features."""
        x, y = split_features_and_labels(small_dataframe, dtype=np.float64)

        assert x.dtype == np.float64
        assert y.dtype == np.int64  # labels always int64

    def test_large_dataframe(self, sample_dataframe):
        """Test with a larger dataframe (100 samples, 784 features)."""
        x, y = split_features_and_labels(sample_dataframe)

        assert x.shape == (100, 784)
        assert y.shape == (100,)


# split_validation_subset


class TestSplitValidationSubset:
    """Test the split_validation_subset function."""

    def test_basic_split_numpy(self):
        """Test basic train/val split with numpy arrays."""
        x = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        x_train, y_train, x_val, y_val = split_validation_subset(x, y, factor=0.2)

        assert x_train.shape[0] == 80
        assert x_val.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_val.shape[0] == 20

    def test_basic_split_tensor(self):
        """Test split with Nova tensors (should return tensors)."""
        x = nova.randn(100, 10)
        y = nova.randint(0, 2, (100,))

        x_train, y_train, x_val, y_val = split_validation_subset(x, y, factor=0.2)

        assert isinstance(x_train, nova.Tensor)
        assert isinstance(y_train, nova.Tensor)
        assert isinstance(x_val, nova.Tensor)
        assert isinstance(y_val, nova.Tensor)
        assert x_train.shape[0] + x_val.shape[0] == 100

    def test_no_data_leakage(self):
        """Test that train and val sets do not share indices."""
        x = np.arange(200).reshape(100, 2).astype(np.float64)
        y = np.random.randint(0, 2, 100)

        x_train, _, x_val, _ = split_validation_subset(
            x, y, factor=0.2, random_state=42
        )

        # Convert to sets of tuples for comparison
        train_set = set(map(tuple, x_train))
        val_set = set(map(tuple, x_val))

        assert len(train_set & val_set) == 0

    def test_reproducibility_with_random_state(self):
        """Test that same random_state produces same split."""
        x = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)

        split1 = split_validation_subset(x, y, factor=0.2, random_state=42)
        split2 = split_validation_subset(x, y, factor=0.2, random_state=42)

        np.testing.assert_array_equal(split1[0], split2[0])
        np.testing.assert_array_equal(split1[2], split2[2])

    def test_stratification_preserves_distribution(self):
        """Test that stratify=True preserves class ratios approximately."""
        # Imbalanced dataset: 80% class 0, 20% class 1
        y = np.array([0] * 80 + [1] * 20)
        x = np.random.rand(100, 5)

        _, y_train, _, y_val = split_validation_subset(
            x, y, factor=0.2, stratify=True, random_state=42
        )

        # Check that class 1 ratio is preserved (~20%) in both sets
        train_ratio = np.mean(y_train == 1)
        val_ratio = np.mean(y_val == 1)

        assert abs(train_ratio - 0.2) < 0.05
        assert abs(val_ratio - 0.2) < 0.1

    def test_invalid_factor_zero(self):
        """Test that factor=0 raises ValueError."""
        x = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        with pytest.raises(ValueError, match="factor must be between 0 and 1"):
            split_validation_subset(x, y, factor=0.0)

    def test_invalid_factor_one(self):
        """Test that factor=1 raises ValueError."""
        x = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        with pytest.raises(ValueError, match="factor must be between 0 and 1"):
            split_validation_subset(x, y, factor=1.0)

    def test_invalid_factor_negative(self):
        """Test that negative factor raises ValueError."""
        x = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        with pytest.raises(ValueError, match="factor must be between 0 and 1"):
            split_validation_subset(x, y, factor=-0.5)

    def test_feature_dimensions_preserved(self):
        """Test that feature dimensions are unchanged after split."""
        x = np.random.rand(200, 784)
        y = np.random.randint(0, 10, 200)

        x_train, _, x_val, _ = split_validation_subset(x, y, factor=0.25)

        assert x_train.shape[1] == 784
        assert x_val.shape[1] == 784


# _validate_dataframe


class TestValidateDataframe:
    """Test the _validate_dataframe internal function."""

    def test_valid_dataframe(self, small_dataframe):
        """Test that a valid DataFrame passes without error."""
        _validate_dataframe(small_dataframe)

    def test_none_dataframe(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="Cannot save empty DataFrame"):
            _validate_dataframe(None)

    def test_empty_dataframe(self):
        """Test that an empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="Cannot save empty DataFrame"):
            _validate_dataframe(pd.DataFrame())

    def test_min_rows_check(self):
        """Test that min_rows constraint is enforced."""
        df = pd.DataFrame({"a": [1], "b": [2]})

        with pytest.raises(ValueError, match="minimum 5 required"):
            _validate_dataframe(df, min_rows=5)

    def test_all_null_columns(self):
        """Test that all-null columns are detected."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})

        with pytest.raises(ValueError, match="all-null columns"):
            _validate_dataframe(df)


# save_to_csv / save_to_parquet / save_to_excel


class TestSaveFunctions:
    """Test CSV, Parquet, and Excel save functions."""

    def test_save_to_csv(self, small_dataframe, tmp_output):
        """Test basic CSV save and verify content."""
        path = tmp_output / "test.csv"
        save_to_csv(small_dataframe, root=path)

        assert path.exists()
        assert path.stat().st_size > 0

        # Reload and compare
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(loaded, small_dataframe)

    def test_save_to_parquet(self, small_dataframe, tmp_output):
        """Test basic Parquet save and verify content."""
        path = tmp_output / "test.parquet"
        save_to_parquet(small_dataframe, root=path)

        assert path.exists()
        assert path.stat().st_size > 0

        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(loaded, small_dataframe)

    def test_save_to_excel(self, small_dataframe, tmp_output):
        """Test basic Excel save and verify content."""
        path = tmp_output / "test.xlsx"
        save_to_excel(small_dataframe, root=path)

        assert path.exists()
        assert path.stat().st_size > 0

        loaded = pd.read_excel(path)
        pd.testing.assert_frame_equal(loaded, small_dataframe)

    def test_save_csv_creates_directories(self, small_dataframe, tmp_output):
        """Test that nested directories are created automatically."""
        path = tmp_output / "a" / "b" / "c" / "test.csv"
        save_to_csv(small_dataframe, root=path)

        assert path.exists()

    def test_save_parquet_creates_directories(self, small_dataframe, tmp_output):
        """Test that nested directories are created for parquet."""
        path = tmp_output / "nested" / "dir" / "test.parquet"
        save_to_parquet(small_dataframe, root=path)

        assert path.exists()

    def test_save_empty_dataframe_raises(self, tmp_output):
        """Test that saving an empty DataFrame raises SaveError."""
        path = tmp_output / "empty.csv"

        with pytest.raises(SaveError):
            save_to_csv(pd.DataFrame(), root=path)

    def test_save_excel_row_limit(self, tmp_output):
        """Test that exceeding Excel's row limit raises SaveError."""
        # Create a DataFrame that claims to exceed 1,048,576 rows
        # We mock len() to avoid actually creating that much data
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_output / "big.xlsx"

        with patch.object(pd.DataFrame, "__len__", return_value=1_048_577):
            with pytest.raises(SaveError):
                save_to_excel(df, root=path)

    def test_save_cleanup_on_failure(self, tmp_output):
        """Test that partial files are cleaned up on save failure."""
        path = tmp_output / "fail.csv"

        with pytest.raises(SaveError):
            save_to_csv(pd.DataFrame(), root=path)

        # Partial file should have been removed
        assert not path.exists()


# _save_dataframe (dispatcher)


class TestSaveDataframe:
    """Test the _save_dataframe dispatcher function."""

    def test_dispatch_csv(self, small_dataframe, tmp_output):
        """Test that format='csv' routes to save_to_csv."""
        path = tmp_output / "dispatch.csv"
        _save_dataframe(small_dataframe, path, format="csv")

        assert path.exists()

    def test_dispatch_parquet(self, small_dataframe, tmp_output):
        """Test that format='parquet' routes to save_to_parquet."""
        path = tmp_output / "dispatch.parquet"
        _save_dataframe(small_dataframe, path, format="parquet")

        assert path.exists()

    def test_dispatch_excel(self, small_dataframe, tmp_output):
        """Test that format='xlsx' routes to save_to_excel."""
        path = tmp_output / "dispatch.xlsx"
        _save_dataframe(small_dataframe, path, format="xlsx")

        assert path.exists()

    def test_unsupported_format_raises(self, small_dataframe, tmp_output):
        """Test that an unsupported format raises ValueError."""
        path = tmp_output / "bad.json"

        with pytest.raises(ValueError, match="Unsupported format"):
            _save_dataframe(small_dataframe, path, format="json")


# split_validation_dataset


class TestSplitValidationDataset:
    """Test the split_validation_dataset function (split + save)."""

    def test_basic_split_and_save_csv(self, sample_dataframe, tmp_output):
        """Test that split produces correct sizes and saves CSV files."""
        train, val = split_validation_dataset(
            sample_dataframe,
            label="label",
            factor=0.2,
            root=tmp_output,
            save_method="csv",
            set_name="train",
            val_name="val",
            random_state=42,
        )

        assert len(train) + len(val) == len(sample_dataframe)
        assert (tmp_output / "train.csv").exists()
        assert (tmp_output / "val.csv").exists()

    def test_basic_split_and_save_parquet(self, sample_dataframe, tmp_output):
        """Test split with parquet output."""
        train, val = split_validation_dataset(
            sample_dataframe,
            label="label",
            factor=0.2,
            root=tmp_output,
            save_method="parquet",
            set_name="train",
            val_name="val",
            random_state=42,
        )

        assert (tmp_output / "train.parquet").exists()
        assert (tmp_output / "val.parquet").exists()
        assert len(train) + len(val) == 100

    def test_basic_split_and_save_excel(self, sample_dataframe, tmp_output):
        """Test split with excel output."""
        train, val = split_validation_dataset(
            sample_dataframe,
            label="label",
            factor=0.2,
            root=tmp_output,
            save_method="excel",
            set_name="train",
            val_name="val",
            random_state=42,
        )

        assert (tmp_output / "train.xlsx").exists()
        assert (tmp_output / "val.xlsx").exists()

    def test_stratification(self, tmp_output):
        """Test that stratify=True preserves class distribution."""
        # Imbalanced: 70 class 0, 30 class 1
        df = pd.DataFrame(
            {
                "label": [0] * 70 + [1] * 30,
                "feat": np.random.rand(100),
            }
        )

        train, val = split_validation_dataset(
            df,
            label="label",
            factor=0.2,
            root=tmp_output,
            save_method="csv",
            stratify=True,
            random_state=42,
        )

        train_ratio = (train["label"] == 1).mean()
        val_ratio = (val["label"] == 1).mean()

        assert abs(train_ratio - 0.3) < 0.05
        assert abs(val_ratio - 0.3) < 0.1

    def test_invalid_factor(self, sample_dataframe, tmp_output):
        """Test that invalid factor raises ValueError."""
        with pytest.raises(ValueError, match="factor must be between 0 and 1"):
            split_validation_dataset(
                sample_dataframe, label="label", factor=0.0, root=tmp_output
            )

    def test_missing_label_column(self, sample_dataframe, tmp_output):
        """Test that a missing label column raises ValueError."""
        with pytest.raises(ValueError, match="Label columns not found"):
            split_validation_dataset(
                sample_dataframe, label="nonexistent", root=tmp_output
            )

    def test_invalid_save_method(self, sample_dataframe, tmp_output):
        """Test that an invalid save_method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid save_method"):
            split_validation_dataset(
                sample_dataframe,
                label="label",
                factor=0.2,
                root=tmp_output,
                save_method="json",
            )

    def test_empty_dataframe_raises(self, tmp_output):
        """Test that an empty DataFrame raises ValueError."""
        with pytest.raises(ValueError):
            split_validation_dataset(
                pd.DataFrame(), label="label", factor=0.2, root=tmp_output
            )

    def test_reproducibility(self, sample_dataframe, tmp_output):
        """Test that same random_state produces the same split."""
        out1 = tmp_output / "run1"
        out2 = tmp_output / "run2"

        train1, val1 = split_validation_dataset(
            sample_dataframe,
            label="label",
            factor=0.2,
            root=out1,
            save_method="csv",
            random_state=42,
        )
        train2, val2 = split_validation_dataset(
            sample_dataframe,
            label="label",
            factor=0.2,
            root=out2,
            save_method="csv",
            random_state=42,
        )

        pd.testing.assert_frame_equal(
            train1.reset_index(drop=True), train2.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            val1.reset_index(drop=True), val2.reset_index(drop=True)
        )


# _check_root


class TestCheckRoot:
    """Test the _check_root directory creation helper."""

    def test_creates_parent_directories(self, tmp_output):
        """Test that parent directories are created."""
        path = tmp_output / "a" / "b" / "c" / "file.txt"
        _check_root(path)

        assert path.parent.exists()

    def test_existing_directory_no_error(self, tmp_output):
        """Test that an already existing directory does not raise."""
        path = tmp_output / "file.txt"
        _check_root(path)


# _validate_idx_file


class TestValidateIdxFile:
    """Test IDX file magic number validation."""

    def test_valid_image_file(self, tmp_output):
        """Test validation of a correct IDX image file."""
        path = tmp_output / "images.gz"
        _make_idx_images_gz(path, n=5)

        assert _validate_idx_file(path, expected_magic=2051)

    def test_valid_label_file(self, tmp_output):
        """Test validation of a correct IDX label file."""
        path = tmp_output / "labels.gz"
        _make_idx_labels_gz(path, n=5)

        assert _validate_idx_file(path, expected_magic=2049)

    def test_wrong_magic_number(self, tmp_output):
        """Test that a wrong magic number returns False."""
        path = tmp_output / "labels.gz"
        _make_idx_labels_gz(path, n=5)

        # Expect image magic but file has label magic
        assert not _validate_idx_file(path, expected_magic=2051)

    def test_nonexistent_file(self, tmp_output):
        """Test that a nonexistent file returns False."""
        path = tmp_output / "nonexistent.gz"

        assert not _validate_idx_file(path, expected_magic=2051)

    def test_corrupted_file(self, tmp_output):
        """Test that a corrupted (non-gzip) file returns False."""
        path = tmp_output / "corrupt.gz"
        path.write_bytes(b"this is not a gzip file")

        assert not _validate_idx_file(path, expected_magic=2051)


# _read_images / _read_labels


class TestReadIdxFiles:
    """Test IDX binary file readers."""

    def test_read_images_shape_and_dtype(self, tmp_output):
        """Test that _read_images returns correct shape and dtype."""
        n = 20
        path = tmp_output / "images.gz"
        _make_idx_images_gz(path, n=n)

        images = _read_images(path)

        assert images.shape == (n, 28, 28)
        assert images.dtype == np.uint8

    def test_read_labels_shape_and_dtype(self, tmp_output):
        """Test that _read_labels returns correct shape and dtype."""
        n = 20
        path = tmp_output / "labels.gz"
        _make_idx_labels_gz(path, n=n)

        labels = _read_labels(path)

        assert labels.shape == (n,)
        assert labels.dtype == np.uint8

    def test_read_images_wrong_magic_raises(self, tmp_output):
        """Test that an image file with wrong magic raises ValueError."""
        path = tmp_output / "bad_images.gz"
        # Write with label magic (2049) instead of image magic (2051)
        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">IIII", 2049, 5, 28, 28))
            f.write(np.zeros(5 * 28 * 28, dtype=np.uint8).tobytes())

        with pytest.raises(ValueError, match="Invalid magic number for images"):
            _read_images(path)

    def test_read_labels_wrong_magic_raises(self, tmp_output):
        """Test that a label file with wrong magic raises ValueError."""
        path = tmp_output / "bad_labels.gz"
        # Write with image magic (2051) instead of label magic (2049)
        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">II", 2051, 5))
            f.write(np.zeros(5, dtype=np.uint8).tobytes())

        with pytest.raises(ValueError, match="Invalid magic number for labels"):
            _read_labels(path)

    def test_read_images_non_28x28_raises(self, tmp_output):
        """Test that non-28x28 image dimensions raise ValueError."""
        path = tmp_output / "bad_dim.gz"
        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">IIII", 2051, 5, 32, 32))  # 32x32 != 28x28
            f.write(np.zeros(5 * 32 * 32, dtype=np.uint8).tobytes())

        with pytest.raises(ValueError, match="Expected 28x28 images"):
            _read_images(path)

    def test_read_images_truncated_data_raises(self, tmp_output):
        """Test that truncated pixel data raises ValueError."""
        path = tmp_output / "truncated.gz"
        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">IIII", 2051, 10, 28, 28))
            # Write only half the expected data
            f.write(np.zeros(10 * 28 * 28 // 2, dtype=np.uint8).tobytes())

        with pytest.raises(ValueError, match="Expected .* bytes"):
            _read_images(path)

    def test_read_labels_truncated_data_raises(self, tmp_output):
        """Test that truncated label data raises ValueError."""
        path = tmp_output / "truncated_labels.gz"
        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">II", 2049, 10))
            f.write(np.zeros(5, dtype=np.uint8).tobytes())  # Only 5 instead of 10

        with pytest.raises(ValueError, match="Expected 10 labels"):
            _read_labels(path)

    def test_read_corrupted_gzip_raises(self, tmp_output):
        """Test that a non-gzip file raises ValueError."""
        path = tmp_output / "not_gzip.gz"
        path.write_bytes(b"not a gzip file at all")

        with pytest.raises(ValueError, match="not a valid gzip file"):
            _read_images(path)

        with pytest.raises(ValueError, match="not a valid gzip file"):
            _read_labels(path)

    def test_read_images_pixel_values_preserved(self, tmp_output):
        """Test that pixel values are correctly preserved through write/read."""
        n = 3
        path = tmp_output / "images.gz"
        pixels = np.arange(n * 28 * 28, dtype=np.uint8).reshape(n, 28, 28) % 255

        with gzip.open(path, "wb") as f:
            f.write(struct.pack(">IIII", 2051, n, 28, 28))
            f.write(pixels.tobytes())

        result = _read_images(path)
        np.testing.assert_array_equal(result, pixels)


# _download_with_retry


class TestDownloadWithRetry:
    """Test the download retry logic."""

    def test_successful_download(self, tmp_output):
        """Test that a successful response saves the file."""
        dest = tmp_output / "file.gz"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "10"}
        mock_response.iter_content = MagicMock(return_value=[b"0123456789"])
        mock_response.raise_for_status = MagicMock()

        with patch(
            "nova.utils.data.preprocessing.requests.get", return_value=mock_response
        ):
            _download_with_retry("http://example.com/file.gz", dest, max_retries=1)

        assert dest.exists()
        assert dest.read_bytes() == b"0123456789"

    def test_all_retries_exhausted_raises(self, tmp_output):
        """Test that DatasetDownloadError is raised after all retries fail."""
        dest = tmp_output / "fail.gz"

        with patch(
            "nova.utils.data.preprocessing.requests.get",
            side_effect=requests.exceptions.ConnectionError("connection error"),
        ):
            with pytest.raises(DatasetDownloadError, match="Failed to download"):
                _download_with_retry(
                    "http://example.com/fail.gz",
                    dest,
                    max_retries=2,
                    backoff_factor=0.01,
                )

    def test_empty_download_retries(self, tmp_output):
        """Test that an empty downloaded file triggers a retry."""
        dest = tmp_output / "empty.gz"
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.iter_content = MagicMock(return_value=[])  # No data
        mock_response.raise_for_status = MagicMock()

        with patch(
            "nova.utils.data.preprocessing.requests.get", return_value=mock_response
        ):
            with pytest.raises(DatasetDownloadError):
                _download_with_retry(
                    "http://example.com/empty.gz",
                    dest,
                    max_retries=2,
                    backoff_factor=0.01,
                )


# download_dataset


class TestDownloadDataset:
    """Test the download_dataset orchestration function."""

    def test_unsupported_dataset_raises(self, tmp_output):
        """Test that an unsupported dataset name raises ValueError."""
        with pytest.raises(ValueError, match="not yet supported"):
            download_dataset("cifar10", root=tmp_output)

    def test_unsupported_format_raises(self, tmp_output):
        """Test that an unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            download_dataset("mnist", root=tmp_output, format="json")

    def test_skips_existing_files(self, tmp_output):
        """Test that existing output files are skipped (no download)."""
        # Pre-create the expected output files
        mnist_dir = tmp_output / "Mnist"
        mnist_dir.mkdir()
        (mnist_dir / "mnist_train.parquet").write_bytes(b"fake")
        (mnist_dir / "mnist_test.parquet").write_bytes(b"fake")

        with patch("nova.utils.data.preprocessing._download_with_retry") as mock_dl:
            download_dataset(
                "mnist", root=tmp_output, format="parquet", force_redownload=False
            )

            # Download should NOT have been called
            mock_dl.assert_not_called()

    def test_force_redownload_does_not_skip(self, tmp_output):
        """Test that force_redownload=True re-downloads even if files exist."""
        mnist_dir = tmp_output / "Mnist"
        mnist_dir.mkdir()
        (mnist_dir / "mnist_train.parquet").write_bytes(b"fake")
        (mnist_dir / "mnist_test.parquet").write_bytes(b"fake")

        # Mock download + read to prevent actual network calls
        with patch("nova.utils.data.preprocessing._download_with_retry") as mock_dl:
            with patch(
                "nova.utils.data.preprocessing._validate_idx_file", return_value=True
            ):
                with patch(
                    "nova.utils.data.preprocessing._read_images",
                    return_value=np.zeros((60000, 28, 28), dtype=np.uint8),
                ):
                    with patch(
                        "nova.utils.data.preprocessing._read_labels",
                        return_value=np.zeros(60000, dtype=np.uint8),
                    ):
                        download_dataset(
                            "mnist",
                            root=tmp_output,
                            format="parquet",
                            force_redownload=True,
                        )

            # Download SHOULD have been called
            assert mock_dl.called

    def test_fashion_mnist_folder_structure(self, tmp_output):
        """Test that fashion-mnist creates the correct directory."""
        # Pre-create files to skip download
        fashion_dir = tmp_output / "FashionMnist"
        fashion_dir.mkdir()
        (fashion_dir / "fashion-mnist_train.csv").write_bytes(b"fake")
        (fashion_dir / "fashion-mnist_test.csv").write_bytes(b"fake")

        with patch("nova.utils.data.preprocessing._download_with_retry"):
            download_dataset("fashion-mnist", root=tmp_output, format="csv")

        assert fashion_dir.exists()
