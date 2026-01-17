import pytest
import nova
import numpy as np
from nova._internal._binding import native_yaml, bootstrap_to
from nova._internal._generators import (
    make_forward_func,
    make_reverse_func,
    make_inplace_func,
    make_method,
)

nova.manual_seed(8)


class TestNativeYAML:
    """Test YAML loading functionality."""

    def test_yaml_loads_successfully(self):
        """Test that YAML file loads without errors."""
        config = native_yaml()
        assert config is not None
        assert "ops" in config
        assert len(config["ops"]) > 0

    def test_yaml_structure(self):
        """Test that YAML has correct structure."""
        config = native_yaml()

        # Check first operation has required fields
        first_op = config["ops"][0]
        assert "name" in first_op
        assert "tensor" in first_op

    def test_yaml_operation_definitions(self):
        """Test that common operations are defined."""
        from nova.utils.decorators.registry import _OPS_REGISTERED

        op_names = list(_OPS_REGISTERED)

        # Check essential operations are present
        assert "add" in op_names
        assert "sub" in op_names
        assert "mul" in op_names
        assert "div" in op_names
        assert "matmul" in op_names


class TestMakeForwardFunc:
    """Test forward function generator."""

    def test_binary_operation(self):
        """Test generating binary operation like __add__."""
        from nova.autograd._ops import Add

        add_func = make_forward_func(Add, raw=False, is_unary=False)

        x = nova.tensor([1.0, 2.0])
        y = nova.tensor([3.0, 4.0])
        result = add_func(x, y)

        assert isinstance(result, nova.Tensor)
        assert nova.allclose(result, [4.0, 6.0])

    def test_unary_operation(self):
        """Test generating unary operation like __neg__."""
        from nova.autograd._ops import Neg

        neg_func = make_forward_func(Neg, raw=False, is_unary=True)

        x = nova.tensor([1.0, -2.0, 3.0])
        result = neg_func(x)

        assert nova.allclose(result, [-1.0, 2.0, -3.0])

    def test_auto_tensor_conversion(self):
        """Test that scalars are auto-converted to tensors."""
        from nova.autograd._ops import Add

        add_func = make_forward_func(Add, raw=False, is_unary=False)

        x = nova.tensor([1.0, 2.0])
        result = add_func(x, 5.0)  # Scalar should be converted

        assert nova.allclose(result, [6.0, 7.0])

    def test_raw_args_no_conversion(self):
        """Test that raw=True prevents automatic conversion."""
        from nova.autograd._ops import GetItem

        getitem_func = make_forward_func(GetItem, raw=True, is_unary=False)

        x = nova.tensor([1.0, 2.0, 3.0])
        result = getitem_func(x, 1)  # Index stays as int

        assert result.item() == 2.0

    def test_binary_requires_argument(self):
        """Test that binary operations require an argument."""
        from nova.autograd._ops import Add

        add_func = make_forward_func(Add, raw=False, is_unary=False)

        x = nova.tensor([1.0, 2.0])
        with pytest.raises(TypeError):
            add_func(x)  # No argument provided


class TestMakeReverseFunc:
    """Test reverse function generator."""

    def test_reverse_operation(self):
        """Test reverse operation like __radd__."""
        from nova.autograd._ops import Sub

        rsub_func = make_reverse_func(Sub)

        x = nova.tensor([1.0, 2.0])
        y = nova.tensor([10.0, 20.0])
        result = rsub_func(x, y)  # Should compute y - x

        assert nova.allclose(result, [9.0, 18.0])


class TestMakeMethod:
    """Test regular method generator."""

    def test_method_generation(self):
        """Test generating regular methods like .add()."""
        from nova.autograd._ops import Add

        add_method = make_method(Add)

        x = nova.tensor([1.0, 2.0])
        y = nova.tensor([3.0, 4.0])
        result = add_method(x, y)

        assert nova.allclose(result, [4.0, 6.0])

    def test_method_with_kwargs(self):
        """Test method with keyword arguments."""
        from nova.autograd._ops import Sum

        sum_method = make_method(Sum)

        x = nova.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = sum_method(x, dim=0)

        assert nova.allclose(result, [4.0, 6.0])


class TestMakeInplaceFunc:
    """Test in-place function generator."""

    def test_inplace_operation(self):
        """Test basic in-place operation."""
        from nova.autograd._ops import Add

        add_inplace = make_inplace_func(
            Add, raw=False, op_name="AddBackward", is_unary=False
        )

        x = nova.tensor([1.0, 2.0], requires_grad=False)
        result = add_inplace(x, 5.0)

        # Should modify in-place
        assert result is x
        assert nova.allclose(x, [6.0, 7.0])

    def test_inplace_requires_grad_error(self):
        """Test that in-place fails on tensors requiring grad."""
        from nova.autograd._ops import Add

        add_inplace = make_inplace_func(
            Add, raw=False, op_name="AddBackward", is_unary=False
        )

        x = nova.tensor([1.0, 2.0], requires_grad=True)

        with pytest.raises(RuntimeError, match="Cannot perform inplace operation"):
            add_inplace(x, 5.0)

    def test_inplace_dtype_preservation(self):
        """Test that in-place preserves original dtype."""
        from nova.autograd._ops import Add

        add_inplace = make_inplace_func(
            Add, raw=False, op_name="AddBackward", is_unary=False
        )

        x = nova.tensor([1, 2], dtype=nova.int)
        add_inplace(x, 5)

        assert x.dtype == nova.int
        assert np.array_equal(x.data, [6, 7])

    def test_inplace_unary_operation(self):
        """Test unary in-place operation."""
        from nova.autograd._ops import Abs

        abs_inplace = make_inplace_func(
            Abs, raw=False, op_name="AbsBackward", is_unary=True
        )

        x = nova.tensor([-1.0, -2.0, 3.0], requires_grad=False)
        result = abs_inplace(x)

        assert result is x
        assert nova.allclose(x, [1.0, 2.0, 3.0])


class TestBootstrapTo:
    """Test the complete bootstrapping process."""

    def test_methods_are_bound(self):
        """Test that methods are successfully bound to Tensor."""
        # These should all exist after bootstrapping
        x = nova.tensor([1.0, 2.0])

        assert hasattr(x, "__add__")
        assert hasattr(x, "__radd__")
        assert hasattr(x, "add")
        assert hasattr(x, "add_")
        assert hasattr(x, "__iadd__")

    def test_dunder_methods_work(self):
        """Test that bound dunder methods work correctly."""
        x = nova.tensor([1.0, 2.0])
        y = nova.tensor([3.0, 4.0])

        # __add__
        result = x + y
        assert nova.allclose(result, [4.0, 6.0])

        # __sub__
        result = x - y
        assert nova.allclose(result, [-2.0, -2.0])

        # __mul__
        result = x * y
        assert nova.allclose(result, [3.0, 8.0])

    def test_reverse_methods_work(self):
        """Test that reverse operations work."""
        x = nova.tensor([1.0, 2.0])

        # __radd__ should be called when left operand is scalar
        result = 5.0 + x
        assert nova.allclose(result, [6.0, 7.0])

        # __rsub__
        result = 10.0 - x
        assert nova.allclose(result, [9.0, 8.0])

    def test_regular_methods_work(self):
        """Test that regular methods work."""
        x = nova.tensor([1.0, 2.0])

        # .add()
        result = x.add(5.0)
        assert nova.allclose(result, [6.0, 7.0])

        # .mul()
        result = x.mul(2.0)
        assert nova.allclose(result, [2.0, 4.0])

    def test_inplace_methods_work(self):
        """Test that in-place methods work."""
        x = nova.tensor([1.0, 2.0], requires_grad=False)

        # add_()
        x.add_(5.0)
        assert nova.allclose(x, [6.0, 7.0])

        # mul_()
        x.mul_(2.0)
        assert nova.allclose(x, [12.0, 14.0])

    def test_unary_operations_work(self):
        """Test that unary operations work."""
        x = nova.tensor([1.0, -2.0, 3.0])

        # __neg__
        result = -x
        assert nova.allclose(result, [-1.0, 2.0, -3.0])

        # __abs__
        result = abs(x)
        assert nova.allclose(result, [1.0, 2.0, 3.0])

    def test_raw_args_operations(self):
        """Test operations that use raw_args flag."""
        x = nova.tensor([1.0, 2.0, 3.0, 4.0])

        # __getitem__ should keep indices as integers
        result = x[1]
        assert result == 2.0

        # __pow__ should keep exponents as scalars
        result = x**2
        assert nova.allclose(result, [1.0, 4.0, 9.0, 16.0])

    def test_mathematical_functions(self):
        """Test mathematical function bindings."""
        x = nova.tensor([1.0, 4.0, 9.0])

        # sqrt
        result = x.sqrt()
        assert nova.allclose(result, [1.0, 2.0, 3.0])

        # exp
        result = nova.tensor([0.0, 1.0]).exp()
        assert nova.allclose(result, [1.0, np.e], rtol=1e-5)

    def test_reduction_operations(self):
        """Test reduction operation bindings."""
        x = nova.tensor([[1.0, 2.0], [3.0, 4.0]])

        # sum
        result = x.sum()
        assert np.isclose(result, 10.0)

        # mean
        result = x.mean()
        assert np.isclose(result, 2.5)

        # max
        result = x.max()
        assert np.isclose(result, 4.0)

    def test_shape_operations(self):
        """Test shape manipulation operation bindings."""
        x = nova.tensor([[1.0, 2.0], [3.0, 4.0]])

        # reshape
        result = x.reshape(4)
        assert result.shape == (4,)

        # permute
        result = x.permute(1, 0)
        assert result.shape == (2, 2)
        assert nova.allclose(result, [[1.0, 3.0], [2.0, 4.0]])


class TestEdgeCases:
    """Test edge cases in the binding system."""

    def test_method_not_double_bound(self):
        """Test that methods aren't bound twice."""
        # Get method reference
        x = nova.tensor([1.0])
        add_method1 = x.__add__

        # Re-bootstrap shouldn't change existing methods
        from nova import Tensor

        bootstrap_to(Tensor)

        add_method2 = x.__add__
        assert add_method1 == add_method2

    def test_chained_operations(self):
        """Test that operations can be chained."""
        x = nova.tensor([1.0, 2.0])

        result = (x + 1.0) * 2.0 - 3.0
        assert nova.allclose(result, [1.0, 3.0])

    def test_mixed_scalar_tensor_ops(self):
        """Test operations mixing scalars and tensors."""
        x = nova.tensor([1.0, 2.0])

        # Tensor op scalar
        result = x + 5.0
        assert nova.allclose(result, [6.0, 7.0])

        # Scalar op tensor (reverse)
        result = 5.0 + x
        assert nova.allclose(result, [6.0, 7.0])

        # Both should give same result
        assert nova.allclose((x + 5.0), (5.0 + x))
