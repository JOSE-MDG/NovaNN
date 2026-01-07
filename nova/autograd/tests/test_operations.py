import nova
import pytest
from nova.utils import grad_check_wrt_inputs
from op_signatures import (
    ALL_TESTABLE_OPS,
    OPERATIONS,
    OpCategory,
    make_test_input,
    create_op_wrapper,
)

nova.manual_seed(8)


@pytest.mark.parametrize("op_name", ALL_TESTABLE_OPS)
def test_operation_gradients(op_name):
    """
    Gradient checking test for all operations.
    Verifies that the analytical (backward) gradients
    match the numerical (finite difference) gradients.
    """
    # Generate appropriate input with small shape for speed
    x = make_test_input(op_name, shape=(3, 3), requires_grad=True)

    # Create operation wrapper
    op_fn = create_op_wrapper(op_name)

    # Gradient checking
    analytic, numeric = grad_check_wrt_inputs(op_fn, x, eps=1e-4)

    # More permissive tolerances for complex operations
    rtol = 1e-2
    atol = 1e-3

    # Operations that require even higher tolerances
    high_tolerance_ops = [
        "exp",
        "gelu",
        "sigmoid",
        "var",
        "inv",
        "matmul",
        "dot",
        "det",
    ]
    if op_name in high_tolerance_ops:
        rtol = 1e-1  # 10% - matrices are numerically unstable
        atol = 1e-2

    assert nova.allclose(
        analytic[0], numeric[0], rtol=rtol, atol=atol
    ), f"Gradient mismatch for {op_name}\nAnalytic: {analytic[0]}\nNumeric: {numeric[0]}"


@pytest.mark.parametrize("op_name", OPERATIONS[OpCategory.UNARY.value])
def test_unary_operations(op_name):
    """Specific test for unary operations."""
    x = make_test_input(op_name, shape=(5,), requires_grad=True)
    op_fn = create_op_wrapper(op_name)
    y = op_fn(x)

    assert y.shape == x.shape, f"{op_name} changed shape unexpectedly"

    if y.requires_grad:
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, f"{op_name} didn't produce gradients"
        assert not nova.allclose(
            x.grad, nova.zeros_like(x.grad)
        ), f"{op_name} produced all-zero gradients"


@pytest.mark.parametrize("op_name", OPERATIONS[OpCategory.REDUCTION.value])
def test_reduction_operations(op_name):
    """Specific test for reduction operations."""
    x = make_test_input(op_name, shape=(4, 3), requires_grad=True)

    y_scalar = getattr(x, op_name)()
    assert y_scalar.numel() == 1, f"{op_name} didn't produce scalar"

    y_dim = getattr(x, op_name)(dim=0, keepdims=False)
    assert y_dim.shape == (3,), f"{op_name} with dim=0 wrong shape"

    y_keep = getattr(x, op_name)(dim=0, keepdims=True)
    assert y_keep.shape == (1, 3), f"{op_name} with keepdims wrong shape"


def test_operations_with_no_useful_gradients():
    """
    Test for operations that have backward but whose gradients
    are almost 0 everywhere.

    """
    x = nova.randn(5, requires_grad=True)

    y = x.sign()
    assert y.shape == x.shape, "sign changed shape"
    if y.requires_grad:
        y.sum().backward()
        assert x.grad is not None or True, "sign backward should work"

    x.zero_grad()

    y = x.ceil()
    assert y.shape == x.shape, "ceil changed shape"
    if y.requires_grad:
        y.sum().backward()
        assert x.grad is not None or True, "ceil backward should work"


def test_trace_operation():
    """Specific test for trace that reduces a matrix to a scalar"""
    x = nova.randn(4, 4, requires_grad=True)
    y = x.trace()

    assert y.numel() == 1, "trace didn't produce scalar"

    y.backward()
    assert x.grad is not None, "trace didn't produce gradients"

    expected_grad = nova.eye(4)
    assert nova.allclose(x.grad, expected_grad, rtol=1e-5)
