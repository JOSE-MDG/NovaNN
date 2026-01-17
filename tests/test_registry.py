import pytest
import nova
from nova.utils.decorators.registry import (
    registry_class,
    registry_op,
    get_registered_classes,
    _MODULES,
    _OPS_REGISTERED,
)
from nova.autograd.function import Function

nova.manual_seed(8)


class TestRegistryClass:
    """Test class registration for serialization."""

    def test_register_simple_class(self):
        """Test registering a simple class."""

        @registry_class
        class TestClass:
            pass

        # Check registration
        key = (TestClass.__module__, TestClass.__name__)
        assert key in _MODULES
        assert _MODULES[key] is TestClass

        # Cleanup
        del _MODULES[key]

    def test_register_returns_original_class(self):
        """Test that decorator returns the original class unmodified."""

        @registry_class
        class MyClass:
            value = 42

        assert MyClass.value == 42
        assert MyClass.__name__ == "MyClass"

        # Cleanup
        key = (MyClass.__module__, MyClass.__name__)
        del _MODULES[key]

    def test_idempotent_registration(self):
        """Test that re-registering a class is idempotent."""

        @registry_class
        class IdempotentClass:
            pass

        key = (IdempotentClass.__module__, IdempotentClass.__name__)
        first_registration = _MODULES[key]

        # Re-register
        @registry_class
        class IdempotentClass:
            pass

        # Should still be the first one
        assert _MODULES[key] is first_registration

        # Cleanup
        del _MODULES[key]

    def test_get_registered_classes(self):
        """Test retrieving registered classes."""

        @registry_class
        class RetrievableClass:
            pass

        module = RetrievableClass.__module__
        name = RetrievableClass.__name__

        retrieved = get_registered_classes(module, name)
        assert retrieved is RetrievableClass

        # Cleanup
        key = (module, name)
        del _MODULES[key]

    def test_get_unregistered_class_returns_none(self):
        """Test that retrieving unregistered class returns None."""
        result = get_registered_classes("fake.module", "FakeClass")
        assert result is None


class TestRegistryOp:
    """Test operation registration for autograd Functions."""

    def test_register_function(self):
        """Test registering an autograd Function."""

        @registry_op("test_add")
        class TestAdd(Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output, grad_output)

        # Check registration
        assert "test_add" in _OPS_REGISTERED
        assert _OPS_REGISTERED["test_add"] is TestAdd

        # Cleanup
        del _OPS_REGISTERED["test_add"]

    def test_register_returns_original_function(self):
        """Test that decorator returns the original Function class."""

        @registry_op("test_mul")
        class TestMul(Function):
            @staticmethod
            def forward(ctx, x, y):
                return x * y

        assert TestMul.__name__ == "TestMul"
        assert hasattr(TestMul, "forward")
        assert hasattr(TestMul, "backward")

        # Cleanup
        del _OPS_REGISTERED["test_mul"]

    def test_register_non_function_raises_error(self):
        """Test that registering non-Function class raises error."""
        with pytest.raises(ValueError, match="Only Function classes can be registered"):

            @registry_op("invalid_op")
            class NotAFunction:
                pass

    def test_idempotent_op_registration(self):
        """Test that re-registering an operation is idempotent."""

        @registry_op("test_sub")
        class TestSub1(Function):
            @staticmethod
            def forward(ctx, x, y):
                return x - y

        first_registration = _OPS_REGISTERED["test_sub"]

        # Re-register with same name
        @registry_op("test_sub")
        class TestSub2(Function):
            @staticmethod
            def forward(ctx, x, y):
                return x - y

        # Should still be the first one (idempotent)
        assert _OPS_REGISTERED["test_sub"] is first_registration

        # Cleanup
        del _OPS_REGISTERED["test_sub"]

    def test_registered_ops_are_accessible(self):
        """Test that registered operations can be retrieved."""

        @registry_op("test_div")
        class TestDiv(Function):
            @staticmethod
            def forward(ctx, x, y):
                return x / y

        # Should be accessible from the registry
        op_class = _OPS_REGISTERED.get("test_div")
        assert op_class is TestDiv

        # Cleanup
        del _OPS_REGISTERED["test_div"]


class TestRegistryIntegration:
    """Test integration between class and op registries."""

    def test_both_decorators_work_together(self):
        """Test using both decorators on the same class."""

        @registry_class
        @registry_op("test_combined")
        class TestCombined(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output,)

        # Should be in both registries
        assert "test_combined" in _OPS_REGISTERED
        class_key = (TestCombined.__module__, TestCombined.__name__)
        assert class_key in _MODULES

        # Cleanup
        del _OPS_REGISTERED["test_combined"]
        del _MODULES[class_key]
