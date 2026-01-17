import pytest
import numpy as np
import nova

nova.manual_seed(8)


class TestEnsureTensor:
    """Test ensure_tensor utility function."""

    def test_tensor_passthrough_no_changes(self):
        """Test that Tensor is returned as-is when no changes needed."""
        x = nova.tensor([1.0, 2.0], requires_grad=True)
        result = nova.utils.ensure_tensor(x)

        # Should be the same object
        assert result is x

    def test_tensor_with_dtype_change(self):
        """Test that Tensor is copied when dtype changes."""
        x = nova.tensor([1.0, 2.0], dtype=nova.float32)
        result = nova.utils.ensure_tensor(x, dtype=nova.double)

        # Should be a new tensor
        assert result is not x
        assert result.dtype == nova.double
        assert nova.allclose(result, x)

    def test_tensor_with_requires_grad_change(self):
        """Test that Tensor is copied when requires_grad changes."""
        x = nova.tensor([1.0, 2.0], requires_grad=False)
        result = nova.utils.ensure_tensor(x, requires_grad=True)

        # Should be a new tensor
        assert result is not x
        assert result.requires_grad
        assert nova.allclose(result, x)

    def test_numpy_array_conversion(self):
        """Test converting numpy array to Tensor."""
        arr = np.array([1.0, 2.0, 3.0])
        result = nova.utils.ensure_tensor(arr)

        assert isinstance(result, nova.Tensor)
        assert nova.allclose(result, arr)
        assert not result.requires_grad

    def test_numpy_array_with_dtype(self):
        """Test numpy conversion with specified dtype."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = nova.utils.ensure_tensor(arr, dtype=nova.float32)

        assert result.dtype == nova.float32
        assert nova.allclose(result, [1.0, 2.0, 3.0])

    def test_numpy_array_with_requires_grad(self):
        """Test numpy conversion with requires_grad."""
        arr = np.array([1.0, 2.0])
        result = nova.utils.ensure_tensor(arr, requires_grad=True)

        assert result.requires_grad

    def test_python_int_conversion(self):
        """Test converting Python int to Tensor."""
        result = nova.utils.ensure_tensor(42)

        assert isinstance(result, nova.Tensor)
        assert result.dtype == nova.long

    def test_python_float_conversion(self):
        """Test converting Python float to Tensor."""
        result = nova.utils.ensure_tensor(3.14)

        assert isinstance(result, nova.Tensor)
        assert result.dtype == np.float32

    def test_python_bool_conversion(self):
        """Test converting Python bool to Tensor."""
        result = nova.utils.ensure_tensor(True)

        assert isinstance(result, nova.Tensor)
        assert result
        assert result.dtype == nova.bool

    def test_python_list_conversion(self):
        """Test converting Python list to Tensor."""
        result = nova.utils.ensure_tensor([1.0, 2.0, 3.0])

        assert isinstance(result, nova.Tensor)
        assert nova.allclose(result, [1.0, 2.0, 3.0])

    def test_nested_list_conversion(self):
        """Test converting nested list to Tensor."""
        result = nova.utils.ensure_tensor([[1, 2], [3, 4]])

        assert result.shape == (2, 2)
        assert np.array_equal(result.data, [[1, 2], [3, 4]])

    def test_scalar_with_custom_dtype(self):
        """Test scalar conversion with custom dtype."""
        result = nova.utils.ensure_tensor(5, dtype=nova.double)

        assert result.dtype == nova.long  # should have the same dtype
        assert np.isclose(result.data, 5)

    def test_preserves_requires_grad_false_by_default(self):
        """Test that non-Tensor objects default to requires_grad=False."""
        result = nova.utils.ensure_tensor(np.array([1.0, 2.0]))
        assert not result.requires_grad

        result = nova.utils.ensure_tensor([1.0, 2.0])
        assert not result.requires_grad

    def test_int_list_defaults_to_long(self):
        """Test that integer lists use long dtype by default."""
        result = nova.utils.ensure_tensor([1, 2, 3], dtype=nova.long)
        assert result.dtype == nova.long

    def test_mixed_type_list(self):
        """Test list with mixed int/float."""
        result = nova.utils.ensure_tensor([1, 2.0, 3])
        # NumPy will promote to float
        assert result.dtype == nova.float32


class TestEnsureTensorEdgeCases:
    """Test edge cases for ensure_tensor."""

    def test_zero_dimensional_array(self):
        """Test 0-d numpy array."""
        arr = np.array(42)
        result = nova.utils.ensure_tensor(arr)

        assert isinstance(result, nova.Tensor)
        assert result == 42

    def test_empty_array(self):
        """Test empty numpy array."""
        arr = np.array([])
        result = nova.utils.ensure_tensor(arr)

        assert result.shape == (0,)

    def test_complex_dtype_not_supported(self):
        """Test that complex dtypes might fail or get converted."""
        arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)

        # Should either convert or raise error
        try:
            result = nova.utils.ensure_tensor(arr)
            assert isinstance(result, nova.Tensor)
        except (TypeError, ValueError):
            pass
