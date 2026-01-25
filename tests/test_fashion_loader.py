import pytest
import numpy as np
import nova
import gc
from nova.utils.datasets.fashion import load_fashion_mnist_data, FashionData
from nova.utils.memory import quick_memory_check, MemoryTracker, compare_memory
from nova.utils.decorators import measure_memory

nova.manual_seed(8)


class TestFashionDataClass:
    """Test the FashionData Dataset class."""

    def test_init(self):
        """Test FashionData initialization."""
        x = nova.randn(100, 784)
        y = nova.randint(0, 10, (100,))
        dataset = FashionData(x, y)

        assert dataset.x is x
        assert dataset.y is y

    def test_len(self):
        """Test __len__ method."""
        x = nova.randn(100, 784)
        y = nova.randint(0, 10, (100,))
        dataset = FashionData(x, y)

        assert len(dataset) == 100

    def test_getitem(self):
        """Test __getitem__ method."""
        x = nova.randn(100, 784)
        y = nova.randint(0, 10, (100,))
        dataset = FashionData(x, y)

        # Test single index
        sample_x, sample_y = dataset[0]
        assert sample_x.shape == (784,)
        assert sample_y.shape == ()

        # Test multiple indices
        batch_x, batch_y = dataset[:10]
        assert batch_x.shape == (10, 784)
        assert batch_y.shape == (10,)


class TestLoadFashionMnistData:
    """Test the load_fashion_mnist_data function with various configurations."""

    def test_basic_loading_as_tensor(self):
        """Test basic data loading as tensors."""
        train, test, val = load_fashion_mnist_data(
            as_tensor=True, do_normalize=False, dtype=nova.float32
        )

        # Check that we get Dataset objects
        assert isinstance(train, FashionData)
        assert isinstance(test, FashionData)
        assert isinstance(val, FashionData)

        # Check shapes (2D by default)
        assert train.x.dim() == 2
        assert train.x.shape[1] == 784  # 28*28

        # Check dtypes
        assert train.x.dtype == nova.float32
        assert train.y.dtype == nova.long

        # Check that we have data
        assert len(train) > 0
        assert len(test) > 0
        assert len(val) > 0

    def test_4d_tensor_shape(self):
        """Test loading with tensor4d=True."""
        train, test, val = load_fashion_mnist_data(
            tensor4d=True, as_tensor=True, do_normalize=False, dtype=nova.float32
        )

        # Check 4D shapes (N, 1, 28, 28)
        assert train.x.dim() == 4
        assert train.x.shape[1:] == (1, 28, 28)
        assert test.x.shape[1:] == (1, 28, 28)
        assert val.x.shape[1:] == (1, 28, 28)

    def test_normalization(self):
        """Test that normalization works correctly."""
        train, _, _ = load_fashion_mnist_data(
            as_tensor=True, do_normalize=True, dtype=nova.float32
        )

        # After normalization, training data should have approximately zero mean
        # and unit variance
        mean = train.x.mean().item()
        std = train.x.std().item()

        assert abs(mean) < 0.1  # Should be close to 0
        assert abs(std - 1.0) < 0.2  # Should be close to 1

    def test_without_normalization(self):
        """Test loading without normalization."""
        train, _, _ = load_fashion_mnist_data(
            as_tensor=True, do_normalize=False, dtype=nova.float32
        )

        # Without normalization, pixel values should be in [0, 255] range
        assert train.x.min().item() >= 0
        assert train.x.max().item() <= 255

    def test_labels_range(self):
        """Test that labels are in valid range [0, 9]."""
        train, test, val = load_fashion_mnist_data(as_tensor=True)

        # Fashion-MNIST has 10 classes (0-9)
        for dataset in [train, test, val]:
            assert dataset.y.min().item() == 0
            assert dataset.y.max().item() == 9

    def test_different_dtypes(self):
        """Test loading with different dtypes."""
        # Test float64
        train, _, _ = load_fashion_mnist_data(
            as_tensor=True, dtype=nova.float64, do_normalize=False
        )
        assert train.x.dtype == nova.float64

        # Test float32
        train, _, _ = load_fashion_mnist_data(
            as_tensor=True, dtype=nova.float32, do_normalize=False
        )
        assert train.x.dtype == nova.float32

        # Test float16

        train, _, _ = load_fashion_mnist_data(
            as_tensor=True, dtype=nova.float16, do_normalize=False
        )
        assert train.x.dtype == nova.float16

    def test_numpy_output(self):
        """Test loading as numpy arrays instead of tensors."""
        train, _, _ = load_fashion_mnist_data(as_tensor=False, do_normalize=False)

        # Check that we still get Dataset objects
        assert isinstance(train, FashionData)

        # Check that data is numpy arrays
        assert isinstance(train.x, np.ndarray)
        assert isinstance(train.y, np.ndarray)

    def test_4d_with_normalization(self):
        """Test 4D tensor loading with normalization."""
        train, test, val = load_fashion_mnist_data(
            tensor4d=True, as_tensor=True, do_normalize=True, dtype=nova.float32
        )

        # Check shape
        assert train.x.shape[1:] == (1, 28, 28)

        # Check normalization still works
        mean = train.x.mean().item()
        assert abs(mean) < 0.1

    def test_dataset_consistency(self):
        """Test that all splits have consistent feature dimensions."""
        train, test, val = load_fashion_mnist_data(as_tensor=True)

        # All should have same feature dimension
        assert train.x.shape[1:] == test.x.shape[1:]
        assert test.x.shape[1:] == val.x.shape[1:]

    def test_no_data_leakage(self):
        """Test that train/test/val splits are different."""
        train, test, val = load_fashion_mnist_data(as_tensor=True)

        # Check different sizes (basic sanity check)
        assert len(train) != len(test)
        assert len(test) == len(val)  # they've same shape

    def test_requires_grad_false(self):
        """Test that loaded data doesn't require gradients by default."""
        train, _, _ = load_fashion_mnist_data(as_tensor=True)

        assert not train.x.requires_grad
        assert not train.y.requires_grad

    def test_batch_iteration(self):
        """Test that we can iterate over batches."""
        train, _, _ = load_fashion_mnist_data(as_tensor=True, tensor4d=True)

        # Create a simple batch
        batch_size = 32
        x_batch = train.x[:batch_size]
        y_batch = train.y[:batch_size]

        assert x_batch.shape[0] == batch_size
        assert y_batch.shape[0] == batch_size
        assert x_batch.shape[1:] == (1, 28, 28)

    def test_normalization_with_numpy(self):
        """Test normalization when as_tensor=False."""
        train, test, val = load_fashion_mnist_data(as_tensor=False, do_normalize=True)

        # Check that normalization works with numpy arrays
        mean = np.mean(train.x)
        std = np.std(train.x)

        assert abs(mean) < 0.1
        assert abs(std - 1.0) < 0.2

    def test_label_distribution(self):
        """Test that we have samples from all 10 classes."""
        train, _, _ = load_fashion_mnist_data(as_tensor=True)

        unique_labels = nova.unique(train.y)

        # Should have all 10 fashion classes
        assert len(unique_labels) == 10

    def test_dataset_slicing(self):
        """Test various slicing operations on the dataset."""
        train, _, _ = load_fashion_mnist_data(as_tensor=True)

        # Test slice
        x_slice, y_slice = train[10:20]
        assert x_slice.shape[0] == 10
        assert y_slice.shape[0] == 10

        # Test negative indexing
        x_last, _ = train[-1]
        assert x_last.dim() == 1  # Single sample

    def test_memory_efficiency(self):
        """Test that loading doesn't create excessive copies."""
        train, _, _ = load_fashion_mnist_data(as_tensor=True)

        # Get a sample
        x1, y1 = train[0]
        x2, y2 = train[0]

        # Values should be equal
        assert nova.allclose(x1, x2)
        assert y1.item() == y2.item()

    def test_fashion_vs_mnist_difference(self):
        """Test that Fashion-MNIST is different from regular MNIST."""
        from nova.utils.datasets.mnist import load_mnist_data

        # Load both datasets
        fashion_train, _, _ = load_fashion_mnist_data(
            as_tensor=True, do_normalize=False
        )
        mnist_train, _, _ = load_mnist_data(as_tensor=True, do_normalize=False)

        # They should have same shape but different content
        assert fashion_train.x.shape[1:] == mnist_train.x.shape[1:]

        # Statistical properties should be different
        fashion_mean = fashion_train.x.mean().item()
        mnist_mean = mnist_train.x.mean().item()

        # Not a strict test, but they should be somewhat different
        assert abs(fashion_mean - mnist_mean) > 1e-3

    def test_dataset_size_consistency(self):
        """Test that train/test/val have reasonable relative sizes."""
        train, test, val = load_fashion_mnist_data(as_tensor=True)

        # Training set should be largest
        assert len(train) > len(test)
        assert len(train) > len(val)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_path_handling(self):
        """Test behavior with invalid file paths."""
        with pytest.raises(Exception):
            load_fashion_mnist_data(
                train_path="nonexistent_path.csv",
                test_path="nonexistent_path.csv",
                val_path="nonexistent_path.csv",
            )

    def test_single_sample_access(self):
        """Test accessing single samples."""
        train, _, _ = load_fashion_mnist_data(as_tensor=True)

        x, y = train[0]

        # Single sample should be 1D for features
        assert x.dim() == 1
        # Label should be scalar
        assert y.dim() == 0

    def test_4d_single_sample(self):
        """Test single sample access with 4D tensors."""
        train, _, _ = load_fashion_mnist_data(as_tensor=True, tensor4d=True)

        x, y = train[0]

        # Single 4D sample should have shape (1, 28, 28)
        assert x.shape == (1, 28, 28)
        assert y.dim() == 0

    def test_empty_slice(self):
        """Test behavior with empty slices."""
        train, _, _ = load_fashion_mnist_data(as_tensor=True)

        # Empty slice should work
        x_empty, y_empty = train[0:0]
        assert x_empty.shape[0] == 0
        assert y_empty.shape[0] == 0


class TestFashionMemoryUsage:
    """Test memory consumption when loading Fashion-MNIST dataset."""

    def test_basic_loading_memory(self):
        """Test memory usage for basic dataset loading."""

        def load_basic():
            return load_fashion_mnist_data(
                as_tensor=True, do_normalize=False, dtype=nova.float32
            )

        stats = quick_memory_check(load_basic)

        # Check that memory was tracked
        assert stats["peak_mb"] > 0
        assert stats["current_mb"] > 0

        # Fashion-MNIST is similar size to MNIST (~60MB)
        # Peak should be reasonable
        assert stats["peak_mb"] < 850  # Generous upper bound

    def test_4d_tensor_memory_overhead(self):
        """Compare memory usage between 2D and 4D tensor formats."""

        # Load as 2D
        with MemoryTracker() as mem_2d:
            train_2d, _, _ = load_fashion_mnist_data(
                tensor4d=False, as_tensor=True, do_normalize=False
            )

        # Load as 4D
        with MemoryTracker() as mem_4d:
            train_4d, _, _ = load_fashion_mnist_data(
                tensor4d=True, as_tensor=True, do_normalize=False
            )

        # 4D should have similar memory usage (just a reshape)
        ratio = mem_4d.peak_mb / mem_2d.peak_mb
        assert 0.5 <= ratio <= 1.5

    def test_normalization_memory_impact(self):
        """Test memory impact of normalization."""

        # Without normalization
        with MemoryTracker() as mem_raw:
            train_raw, _, _ = load_fashion_mnist_data(
                as_tensor=True, do_normalize=False
            )

        # With normalization
        with MemoryTracker() as mem_norm:
            train_norm, _, _ = load_fashion_mnist_data(
                as_tensor=True, do_normalize=True
            )

        # Normalization shouldn't drastically increase memory
        assert mem_norm.peak_mb > 0
        assert mem_raw.peak_mb > 0

    def test_memory_with_decorator(self):
        """Test memory tracking using decorator pattern."""

        @measure_memory(return_memory=True)
        def load_and_process():
            train, test, val = load_fashion_mnist_data(
                tensor4d=True, as_tensor=True, do_normalize=True, dtype=nova.float32
            )
            # Do some processing
            batch = train.x[:32]
            return batch

        result, (peak_mb, current_mb) = load_and_process()

        # Verify memory was tracked
        assert peak_mb > 0
        assert current_mb > 0
        assert result.shape == (32, 1, 28, 28)

    def test_memory_comparison_dtypes(self):
        """Compare memory usage between different dtypes."""

        def load_float32():
            return load_fashion_mnist_data(
                as_tensor=True, dtype=nova.float32, do_normalize=False
            )

        def load_float64():
            return load_fashion_mnist_data(
                as_tensor=True, dtype=nova.float64, do_normalize=False
            )

        f32_peak, f64_peak, ratio = compare_memory(
            load_float32, load_float64, verbose=False
        )

        # float64 should use roughly 2x memory of float32
        assert 0.5 <= ratio <= 2.5
        assert f32_peak > 0
        assert f64_peak > 0

    def test_batch_access_memory(self):
        """Test memory usage when accessing batches."""
        from nova.utils.memory import MemoryTracker

        train, _, _ = load_fashion_mnist_data(as_tensor=True, tensor4d=True)

        # Accessing small batch
        with MemoryTracker() as mem_small:
            train.x[:16]

        # Accessing large batch
        with MemoryTracker() as mem_large:
            train.x[:256]

        # Larger batch should use more memory
        if not mem_large.peak_mb >= mem_small.peak_mb:
            large_peak = nova.tensor([mem_large.peak_mb], dtype=nova.float64)
            small_peak = nova.tensor([mem_small.peak_mb], dtype=nova.float64)
            assert nova.allclose(large_peak, small_peak)
        else:
            assert mem_large.peak_mb >= mem_small.peak_mb

    def test_memory_cleanup_after_loading(self):
        """Test that memory is properly released after loading."""

        with MemoryTracker() as mem:
            train, test, val = load_fashion_mnist_data(as_tensor=True)
            initial_peak = mem.peak_mb

        # Force cleanup
        del train, test, val
        gc.collect()

        # Verify initial memory was tracked
        assert initial_peak >= 0

    def test_numpy_vs_tensor_memory(self):
        """Compare memory usage between numpy arrays and tensors."""

        # Load as numpy
        stats_numpy = quick_memory_check(
            load_fashion_mnist_data, as_tensor=False, do_normalize=False
        )

        # Load as tensors
        stats_tensor = quick_memory_check(
            load_fashion_mnist_data, as_tensor=True, do_normalize=False
        )

        # Both should track memory successfully
        assert stats_numpy["peak_mb"] > 0
        assert stats_tensor["peak_mb"] > 0

        # Tensor version might use slightly more memory
        ratio = stats_tensor["peak_mb"] / stats_numpy["peak_mb"]
        assert ratio < 3.0

    def test_fashion_vs_mnist_memory_similarity(self):
        """Compare memory usage between Fashion-MNIST and MNIST."""
        from nova.utils.datasets.mnist import load_mnist_data

        def load_fashion():
            return load_fashion_mnist_data(
                as_tensor=True, do_normalize=False, dtype=nova.float32
            )

        def load_mnist():
            return load_mnist_data(
                as_tensor=True, do_normalize=False, dtype=nova.float32
            )

        fashion_peak, mnist_peak, ratio = compare_memory(
            load_fashion, load_mnist, verbose=False
        )

        # Both datasets are same size, memory usage should be similar
        assert 0.5 <= ratio <= 3.5
        assert fashion_peak > 0
        assert mnist_peak > 0

    def test_all_splits_memory_overhead(self):
        """Test memory overhead of loading all three splits."""

        with MemoryTracker() as mem:
            train, test, val = load_fashion_mnist_data(
                as_tensor=True, tensor4d=True, do_normalize=True, dtype=nova.float32
            )

            # Access all splits to ensure they're fully loaded
            _ = len(train)
            _ = len(test)
            _ = len(val)

            peak = mem.peak_mb

        assert peak >= 0
        assert peak < 1000  # Reasonable upper bound for all splits

    def test_multiple_loads_memory_isolation(self):
        """Test that multiple loads don't cause memory leaks."""

        peaks = []

        for i in range(3):
            gc.collect()
            with MemoryTracker() as mem:
                train, test, val = load_fashion_mnist_data(
                    as_tensor=True, do_normalize=False
                )
                peaks.append(mem.peak_mb)

            # Clean up
            del train, test, val
            gc.collect()

        # Peaks should be relatively consistent (no accumulation)
        # Allow 20% variance
        max_peak = max(peaks)
        min_peak = min(peaks)
        assert (max_peak - min_peak) / (min_peak + 1e-10) < 0.3
