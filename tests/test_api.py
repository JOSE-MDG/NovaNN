import pytest
import nova
import numpy as np

nova.manual_seed(42)


# Creation Functions


def test_tensor_creation():
    """Test basic tensor creation."""
    x = nova.tensor([1, 2, 3])
    assert isinstance(x, nova.Tensor)
    assert x.shape == (3,)
    assert np.array_equal(x.data, [1, 2, 3])


def test_zeros():
    """Test zeros creation."""
    x = nova.zeros((3, 4))
    assert x.shape == (3, 4)
    assert nova.all(x == 0)


def test_ones():
    """Test ones creation."""
    x = nova.ones((2, 3))
    assert x.shape == (2, 3)
    assert nova.all(x == 1)


def test_empty():
    """Test empty tensor creation."""
    x = nova.empty((2, 2))
    assert x.shape == (2, 2)


def test_full():
    """Test full tensor creation."""
    x = nova.full((2, 3), 7.0)
    assert x.shape == (2, 3)
    assert nova.all(x == 7.0)


def test_eye():
    """Test identity matrix creation."""
    x = nova.eye(3)
    assert x.shape == (3, 3)
    assert np.array_equal(x.data, np.eye(3))


def test_arange():
    """Test arange creation."""
    x = nova.arange(5)
    assert np.array_equal(x.data, [0, 1, 2, 3, 4])

    y = nova.arange(2, 8, 2)
    assert np.array_equal(y.data, [2, 4, 6])


def test_linspace():
    """Test linspace creation."""
    x = nova.linspace(0, 1, 5)
    assert len(x) == 5
    assert np.isclose(x.data[0], 0.0)
    assert np.isclose(x.data[-1], 1.0)


def test_zeros_like():
    """Test zeros_like creation."""
    x = nova.tensor([[1, 2], [3, 4]])
    y = nova.zeros_like(x)
    assert y.shape == x.shape
    assert nova.all(y == 0)


def test_ones_like():
    """Test ones_like creation."""
    x = nova.tensor([[1, 2], [3, 4]])
    y = nova.ones_like(x)
    assert y.shape == x.shape
    assert nova.all(y == 1)


def test_full_like():
    """Test full_like creation."""
    x = nova.tensor([[1, 2], [3, 4]])
    y = nova.full_like(x, 5.0)
    assert y.shape == x.shape
    assert nova.all(y == 5.0)


# Random Functions


def test_rand():
    """Test uniform random tensor."""
    x = nova.rand(3, 4)
    assert x.shape == (3, 4)
    assert nova.all((x.data >= 0) & (x.data <= 1))


def test_randn():
    """Test normal random tensor."""
    x = nova.randn(100, 100)
    assert x.shape == (100, 100)
    # Mean should be close to 0, std close to 1
    assert nova.abs(x.mean()).item() < 0.2
    assert nova.abs(x.std() - 1.0).item() < 0.2


def test_randint():
    """Test random integer tensor."""
    x = nova.randint(0, 10, size=(5, 5))
    assert x.shape == (5, 5)
    assert nova.all((x.data >= 0) & (x.data < 10))
    assert x.dtype in (nova.int, nova.long)


def test_randperm():
    """Test random permutation."""
    x = nova.randperm(10)
    assert len(x) == 10
    # Should contain all numbers from 0 to 9
    assert set(x.data.tolist()) == set(range(10))


def test_uniform():
    """Test uniform distribution."""
    x = nova.uniform(2.0, 5.0, size=(50,))
    assert nova.all((x.data >= 2.0) & (x.data <= 5.0))


def test_normal():
    """Test normal distribution."""
    x = nova.normal(mean=5.0, std=2.0, size=(100,))
    # Mean should be close to 5.0
    assert abs(x.mean() - 5.0).item() < 1.0


def test_manual_seed():
    """Test manual seed for reproducibility."""
    nova.manual_seed(123)
    x1 = nova.randn(5)

    nova.manual_seed(123)
    x2 = nova.randn(5)

    assert nova.allclose(x1, x2)


# Mathematical Functions


def test_abs():
    """Test absolute value."""
    x = nova.tensor([-1.0, -2.0, 3.0])
    y = nova.abs(x)
    assert nova.allclose(y, [1.0, 2.0, 3.0])


def test_sqrt():
    """Test square root."""
    x = nova.tensor([1.0, 4.0, 9.0])
    y = nova.sqrt(x)
    assert nova.allclose(y, [1.0, 2.0, 3.0])


def test_exp():
    """Test exponential."""
    x = nova.tensor([0.0, 1.0, 2.0])
    y = nova.exp(x)
    assert nova.allclose(y, np.exp([0.0, 1.0, 2.0]))


def test_log():
    """Test natural logarithm."""
    x = nova.tensor([1.0, np.e, np.e**2])
    y = nova.log(x)
    assert nova.allclose(y, [0.0, 1.0, 2.0])


def test_pow():
    """Test power function."""
    x = nova.tensor([2.0, 3.0, 4.0])
    y = nova.pow(x, 2)
    assert nova.allclose(y, [4.0, 9.0, 16.0])


def test_floor():
    """Test floor function."""
    x = nova.tensor([1.5, 2.9, -1.5])
    y = nova.floor(x)
    assert np.array_equal(y.data, [1.0, 2.0, -2.0])


def test_ceil():
    """Test ceiling function."""
    x = nova.tensor([1.5, 2.1, -1.5])
    y = nova.ceil(x)
    assert np.array_equal(y.data, [2.0, 3.0, -1.0])


def test_sign():
    """Test sign function."""
    x = nova.tensor([-2.0, 0.0, 3.0])
    y = nova.sign(x)
    assert np.array_equal(y.data, [-1.0, 0.0, 1.0])


def test_clamp():
    """Test clamp function."""
    x = nova.tensor([-1.0, 0.5, 2.0])
    y = nova.clamp(x, 0.0, 1.0)
    assert nova.allclose(y, [0.0, 0.5, 1.0])


# Trigonometric Functions


def test_sin():
    """Test sine function."""
    x = nova.tensor([0.0, np.pi / 2, np.pi])
    y = nova.sin(x)
    assert nova.allclose(y, [0.0, 1.0, 0.0], atol=1e-6)


def test_cos():
    """Test cosine function."""
    x = nova.tensor([0.0, np.pi / 2, np.pi])
    y = nova.cos(x)
    assert nova.allclose(y, [1.0, 0.0, -1.0], atol=1e-6)


def test_tan():
    """Test tangent function."""
    x = nova.tensor([0.0, np.pi / 4])
    y = nova.tan(x)
    assert nova.allclose(y, [0.0, 1.0], atol=1e-6)


def test_tanh():
    """Test hyperbolic tangent."""
    x = nova.tensor([0.0, 1.0])
    y = nova.tanh(x)
    assert nova.allclose(y, nova.tanh([0.0, 1.0]))


def test_arcsin():
    """Test arcsine function."""
    x = nova.tensor([0.0, 0.5, 1.0])
    y = nova.arcsin(x)
    assert nova.allclose(y, nova.arcsin([0.0, 0.5, 1.0]))


def test_arccos():
    """Test arccosine function."""
    x = nova.tensor([0.0, 0.5, 1.0])
    y = nova.arccos(x)
    assert nova.allclose(y, nova.arccos([0.0, 0.5, 1.0]))


def test_arctan():
    """Test arctangent function."""
    x = nova.tensor([0.0, 1.0])
    y = nova.arctan(x)
    assert nova.allclose(y, nova.arctan([0.0, 1.0]))


def test_sec():
    """Test secant function."""
    x = nova.tensor([0.0])
    y = nova.sec(x)
    assert np.isclose(y.data[0], 1.0)


# Reduction Functions


def test_sum():
    """Test sum reduction."""
    x = nova.tensor([[1, 2], [3, 4]])
    assert nova.sum(x).data == 10

    y = nova.sum(x, dim=0)
    assert nova.allclose(y, nova.tensor([4, 6]))


def test_mean():
    """Test mean reduction."""
    x = nova.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert nova.allclose(nova.mean(x), nova.tensor(2.5))

    y = nova.mean(x, dim=0)
    assert nova.allclose(y, nova.tensor([2.0, 3.0]))


def test_var():
    """Test variance reduction."""
    x = nova.tensor([1.0, 2.0, 3.0, 4.0])
    var = nova.var(x)
    assert np.isclose(var.data, np.var([1.0, 2.0, 3.0, 4.0]))


def test_std():
    """Test standard deviation."""
    x = nova.tensor([1.0, 2.0, 3.0, 4.0])
    std = nova.std(x)
    assert np.isclose(std.data, np.std([1.0, 2.0, 3.0, 4.0]))


def test_max():
    """Test max reduction."""
    x = nova.tensor([[1, 5], [3, 2]])
    assert nova.max(x) == 5

    y = nova.max(x, dim=0)
    assert nova.allclose(y, nova.tensor([3, 5]))


def test_min():
    """Test min reduction."""
    x = nova.tensor([[1, 5], [3, 2]])
    assert nova.min(x).data == 1

    y = nova.min(x, dim=0)
    assert nova.allclose(y, nova.tensor([1, 2]))


def test_maximum():
    """Test element-wise maximum."""
    x = nova.tensor([1, 5, 3])
    y = nova.tensor([2, 3, 4])
    z = nova.maximum(x, y)
    assert nova.allclose(z, nova.tensor([2, 5, 4]))


def test_minimum():
    """Test element-wise minimum."""
    x = nova.tensor([1, 5, 3])
    y = nova.tensor([2, 3, 4])
    z = nova.minimum(x, y)
    assert nova.allclose(z, nova.tensor([1, 3, 3]))


def test_argmax():
    """Test argmax."""
    x = nova.tensor([[1, 5], [3, 2]])
    idx = nova.argmax(x)
    assert idx == 1  # Flattened index

    idx = nova.argmax(x, dim=0)
    # Compare each element since argmax returns array
    assert idx[0] == 1 and idx[1] == 0


def test_argmin():
    """Test argmin."""
    x = nova.tensor([[1, 5], [3, 2]])
    idx = nova.argmin(x)
    assert idx == 0


def test_argsort():
    """Test argsort."""
    x = nova.tensor([3, 1, 4, 2])
    idx = nova.argsort(x)
    # Compare elements individually
    assert list(idx) == [1, 3, 0, 2]


# Linear Algebra


def test_dot():
    """Test dot product."""
    x = nova.tensor([1, 2, 3])
    y = nova.tensor([4, 5, 6])
    z = nova.dot(x, y)
    assert z == 32  # 1*4 + 2*5 + 3*6


def test_det():
    """Test determinant."""
    x = nova.tensor([[1.0, 2.0], [3.0, 4.0]])
    det = nova.det(x)
    assert np.isclose(det.data, -2.0)


def test_inv():
    """Test matrix inverse."""
    x = nova.tensor([[1.0, 2.0], [3.0, 4.0]])
    inv = nova.inv(x)
    # x @ inv should be identity
    result = x @ inv
    assert nova.allclose(result, nova.eye(2), atol=1e-5)


def test_trace():
    """Test matrix trace."""
    x = nova.tensor([[1, 2], [3, 4]])
    tr = nova.trace(x)
    assert tr == 5  # 1 + 4


def test_norm():
    """Test vector norm."""
    x = nova.tensor([3.0, 4.0])
    n = nova.norm(x)
    assert np.isclose(n.data, 5.0)  # sqrt(3^2 + 4^2)


# Shape Manipulation


def test_reshape():
    """Test reshape."""
    x = nova.tensor([[1, 2], [3, 4]])
    y = nova.reshape(x, (4,))
    assert y.shape == (4,)
    assert np.array_equal(y.data, [1, 2, 3, 4])


def test_permute():
    """Test permute (transpose)."""
    x = nova.tensor([[1, 2], [3, 4]])
    y = nova.permute(x, 1, 0)
    assert y.shape == (2, 2)
    expected = nova.tensor([[1, 3], [2, 4]])
    assert nova.allclose(y, expected)


def test_flatten():
    """Test flatten."""
    x = nova.tensor([[1, 2], [3, 4]])
    y = nova.flatten(x)
    assert y.shape == (4,)


def test_unsqueeze():
    """Test unsqueeze (add dimension)."""
    x = nova.tensor([1, 2, 3])
    y = nova.unsqueeze(x, 0)
    assert y.shape == (1, 3)


def test_split():
    """Test split."""
    x = nova.tensor([1, 2, 3, 4, 5, 6])
    chunks = nova.split(x, 3)
    assert len(chunks) == 3
    assert chunks[0].shape == (2,)


def test_tile():
    """Test tile (repeat)."""
    x = nova.tensor([1, 2])
    y = nova.tile(x, (2, 3))
    assert y.shape == (2, 6)


def test_repeat_interleave():
    """Test repeat_interleave."""
    x = nova.tensor([1, 2, 3])
    y = nova.repeat_interleave(x, 2)
    expected = nova.tensor([1, 1, 2, 2, 3, 3])
    assert nova.allclose(y, expected)


def test_pad():
    """Test padding."""
    x = nova.tensor([1, 2, 3])
    y = nova.pad(x, (1, 1), mode="constant")
    expected = nova.tensor([0, 1, 2, 3, 0])
    assert nova.allclose(y, expected)


# Concatenation and Stacking


def test_cat():
    """Test concatenation."""
    x = nova.tensor([[1, 2]])
    y = nova.tensor([[3, 4]])
    z = nova.cat([x, y], dim=0)
    assert z.shape == (2, 2)
    expected = nova.tensor([[1, 2], [3, 4]])
    assert nova.allclose(z, expected)


def test_stack():
    """Test stacking."""
    x = nova.tensor([1, 2])
    y = nova.tensor([3, 4])
    z = nova.stack([x, y], dim=0)
    assert z.shape == (2, 2)
    expected = nova.tensor([[1, 2], [3, 4]])
    assert nova.allclose(z, expected)


# Comparison and Logic


def test_allclose():
    """Test allclose comparison."""
    x = nova.tensor([1.0, 2.0, 3.0])
    y = nova.tensor([1.0, 2.0, 3.0])
    assert nova.allclose(x, y)

    z = nova.tensor([1.0, 2.0, 3.1])
    assert not nova.allclose(x, z, atol=1e-2)


def test_all():
    """Test all reduction."""
    x = nova.tensor([True, True, True])
    assert nova.all(x)

    y = nova.tensor([True, False, True])
    assert not nova.all(y)


def test_any():
    """Test any reduction."""
    x = nova.tensor([False, False, False])
    assert not nova.any(x)

    y = nova.tensor([False, True, False])
    assert nova.any(y)


def test_where():
    """Test where (conditional selection)."""
    condition = nova.tensor([True, False, True])
    x = nova.tensor([1, 2, 3])
    y = nova.tensor([4, 5, 6])
    result = nova.where(condition, x, y)
    expected = nova.tensor([1, 5, 3])
    assert nova.allclose(result, expected)


def test_isnan():
    """Test isnan detection."""
    x = nova.tensor([1.0, float("nan"), 3.0])
    mask = nova.isnan(x)
    assert np.array_equal(mask, [False, True, False])


def test_isinf():
    """Test isinf detection."""
    x = nova.tensor([1.0, float("inf"), 3.0])
    mask = nova.isinf(x)
    assert np.array_equal(mask, [False, True, False])


def test_argwhere():
    """Test argwhere."""
    x = nova.tensor([0, 1, 0, 1, 1])
    indices = nova.argwhere(x)
    assert np.array_equal(indices, [[1], [3], [4]])


def test_unique():
    """Test unique values."""
    x = nova.tensor([1, 2, 2, 3, 3, 3])
    u = nova.unique(x)
    assert np.array_equal(u, [1, 2, 3])


# Utilities


def test_one_hot():
    """Test one-hot encoding."""
    x = nova.tensor([0, 1, 2])
    y = nova.one_hot(x, num_classes=3)
    expected = nova.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert nova.allclose(y, expected)


def test_as_strided():
    """Test as_strided for advanced indexing."""
    x = nova.tensor([1, 2, 3, 4, 5, 6])
    # Create sliding windows
    y = nova.as_strided(x, size=(4, 2), strides=(x.data.strides[0], x.data.strides[0]))
    assert y.shape == (4, 2)


# Gradient Context Managers


def test_no_grad():
    """Test no_grad context manager."""
    assert nova.is_grad_enabled()

    with nova.no_grad():
        assert not nova.is_grad_enabled()
        x = nova.tensor([1.0, 2.0], requires_grad=True)
        y = x * 2
        assert y.grad_fn is None

    assert nova.is_grad_enabled()


def test_enable_grad():
    """Test enable_grad context manager."""
    with nova.no_grad():
        assert not nova.is_grad_enabled()

        with nova.enable_grad():
            assert nova.is_grad_enabled()

    assert nova.is_grad_enabled()


# Dtype Tests


def test_dtypes_available():
    """Test that dtypes are accessible."""
    assert hasattr(nova, "float32")
    assert hasattr(nova, "double")
    assert hasattr(nova, "int")
    assert hasattr(nova, "long")
    assert hasattr(nova, "bool")
    assert hasattr(nova, "uint8")
    assert hasattr(nova, "int8")


def test_dtype_usage():
    """Test using dtypes in tensor creation."""
    x = nova.tensor([1.0, 2.0], dtype=nova.float32)
    assert x.dtype == np.float32

    y = nova.tensor([1, 2], dtype=nova.long)
    assert y.dtype == np.int64

    z = nova.tensor([True, False], dtype=nova.bool)
    assert z.dtype == np.bool_


# Version and Module Info


def test_version_exists():
    """Test that version string exists."""
    assert hasattr(nova, "__version__")
    assert isinstance(nova.__version__, str)


def test_tensor_class_available():
    """Test that Tensor class is accessible."""
    assert hasattr(nova, "Tensor")
    assert isinstance(nova.Tensor, type)
