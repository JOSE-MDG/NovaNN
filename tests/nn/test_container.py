import pytest
import nova
from collections import OrderedDict
from nova.nn import Sequential, Module, Linear, ReLU

nova.manual_seed(8)


class DummyModule(Module):
    """Simple module for testing."""

    def forward(self, x):
        return x + 1


class TestSequentialConstruction:
    """Test Sequential construction and basic operations."""

    def test_construction_with_modules(self):
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        assert len(model) == 3
        assert isinstance(model[0], Linear)
        assert isinstance(model[1], ReLU)

    def test_construction_with_ordered_dict(self):
        layers = OrderedDict(
            [("fc1", Linear(10, 20)), ("relu", ReLU()), ("fc2", Linear(20, 5))]
        )
        model = Sequential(layers)
        assert len(model) == 3
        assert "fc1" in model._modules
        assert "relu" in model._modules

    def test_empty_sequential(self):
        model = Sequential()
        assert len(model) == 0


class TestSequentialForward:
    """Test forward pass through Sequential."""

    def test_forward_chains_modules(self):
        model = Sequential(DummyModule(), DummyModule(), DummyModule())
        x = nova.tensor([1.0])
        y = model(x)
        assert y.item() == 4.0  # 1 + 1 + 1 + 1

    def test_forward_with_linear_layers(self):
        model = Sequential(Linear(10, 20), Linear(20, 5))
        x = nova.randn(4, 10)
        y = model(x)
        assert y.shape == (4, 5)


class TestSequentialIndexing:
    """Test indexing and slicing operations."""

    def test_getitem_by_index(self):
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        assert isinstance(model[0], Linear)
        assert isinstance(model[1], ReLU)
        assert isinstance(model[-1], Linear)

    def test_getitem_by_slice(self):
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 10), ReLU())
        sub = model[0:2]
        assert isinstance(sub, Sequential)
        assert len(sub) == 2

    def test_setitem(self):
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        new_module = DummyModule()
        model[1] = new_module
        assert model[1] is new_module

    def test_delitem(self):
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        del model[1]
        assert len(model) == 2
        assert isinstance(model[1], Linear)  # Renumbered

    def test_delitem_slice(self):
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 10), ReLU())
        del model[1:3]
        assert len(model) == 2

    def test_index_out_of_range(self):
        model = Sequential(Linear(10, 20), ReLU())
        with pytest.raises(IndexError):
            _ = model[5]


class TestSequentialMethods:
    """Test Sequential methods (append, insert, extend, pop)."""

    def test_append(self):
        model = Sequential(Linear(10, 20))
        model.append(ReLU())
        assert len(model) == 2
        assert isinstance(model[-1], ReLU)

    def test_insert(self):
        model = Sequential(Linear(10, 20), Linear(20, 5))
        model.insert(1, ReLU())
        assert len(model) == 3
        assert isinstance(model[1], ReLU)

    def test_insert_negative_index(self):
        model = Sequential(Linear(10, 20), Linear(20, 5))
        model.insert(-1, ReLU())
        assert isinstance(model[-2], ReLU)

    def test_extend(self):
        model = Sequential(Linear(10, 20))
        other = Sequential(ReLU(), Linear(20, 5))
        model.extend(other)
        assert len(model) == 3

    def test_pop(self):
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        popped = model.pop(1)
        assert isinstance(popped, ReLU)
        assert len(model) == 2

    def test_pop_slice(self):
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 10), ReLU())
        popped = model.pop(slice(1, 3))
        assert isinstance(popped, Sequential)
        assert len(popped) == 2
        assert len(model) == 2


class TestSequentialIteration:
    """Test iteration over Sequential."""

    def test_iter(self):
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        modules = list(model)
        assert len(modules) == 3
        assert all(isinstance(m, Module) for m in modules)

    def test_len(self):
        model = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        assert len(model) == 3


class TestSequentialRepr:
    """Test string representation."""

    def test_repr_simple(self):
        model = Sequential(Linear(10, 20), ReLU())
        repr_str = repr(model)
        assert "Sequential" in repr_str
        assert "Linear" in repr_str
        assert "ReLU" in repr_str

    def test_repr_empty(self):
        model = Sequential()
        assert repr(model) == "Sequential()"

    def test_repr_repeated_modules(self):
        model = Sequential(Linear(10, 10), Linear(10, 10), Linear(10, 10))
        repr_str = repr(model)
        assert "3 x" in repr_str or "(0-2)" in repr_str


class TestSequentialArithmetic:
    """Test arithmetic operations (add, mul)."""

    def test_add_two_sequentials(self):
        model1 = Sequential(Linear(10, 20), ReLU())
        model2 = Sequential(Linear(20, 10), ReLU())
        combined = model1 + model2

        assert len(combined) == 4
        assert len(model1) == 2  # Original unchanged
        assert len(model2) == 2  # Original unchanged
        assert isinstance(combined, Sequential)

    def test_add_invalid_type(self):
        model = Sequential(Linear(10, 20))
        with pytest.raises(ValueError, match="add operator supports only"):
            _ = model + "invalid"

    def test_iadd_sequential(self):
        model = Sequential(Linear(10, 20), ReLU())
        other = Sequential(Linear(20, 10))
        original_id = id(model)

        model += other

        assert len(model) == 3
        assert id(model) == original_id  # In-place modification
        assert isinstance(model[2], Linear)

    def test_iadd_invalid_type(self):
        model = Sequential(Linear(10, 20))
        with pytest.raises(ValueError, match="add operator supports only"):
            model += [ReLU()]

    def test_mul_repeat_sequential(self):
        model = Sequential(Linear(10, 10), ReLU())
        repeated = model * 3

        assert len(repeated) == 6
        assert len(model) == 2  # Original unchanged
        assert isinstance(repeated[0], Linear)
        assert isinstance(repeated[1], ReLU)
        assert isinstance(repeated[2], Linear)

    def test_mul_invalid_type(self):
        model = Sequential(Linear(10, 20))
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = model * "3"

    def test_mul_non_positive(self):
        model = Sequential(Linear(10, 20))
        with pytest.raises(ValueError, match="Non-positive multiplication"):
            _ = model * 0
        with pytest.raises(ValueError, match="Non-positive multiplication"):
            _ = model * -1

    def test_rmul_repeat_sequential(self):
        model = Sequential(Linear(10, 10), ReLU())
        repeated = 3 * model

        assert len(repeated) == 6
        assert len(model) == 2  # Original unchanged

    def test_imul_repeat_sequential(self):
        model = Sequential(Linear(10, 10), ReLU())
        original_id = id(model)

        model *= 3

        assert len(model) == 6
        assert id(model) == original_id  # In-place modification
        assert isinstance(model[0], Linear)
        assert isinstance(model[1], ReLU)
        assert isinstance(model[4], Linear)

    def test_imul_invalid_type(self):
        model = Sequential(Linear(10, 20))
        with pytest.raises(TypeError, match="unsupported operand type"):
            model *= "3"

    def test_imul_non_positive(self):
        model = Sequential(Linear(10, 20))
        with pytest.raises(ValueError, match="Non-positive multiplication"):
            model *= 0
