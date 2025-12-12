import pytest
import numpy as np
import novann as nn

RNG = np.random.RandomState(0)


def test_dropout_eval_mode():
    """Test dropout in evaluation mode (no dropout applied)."""

    dropout = nn.Dropout(p=0.5)
    dropout.eval()

    x = np.ones((2, 3), dtype=np.float32) * 2.0
    output = dropout(x)

    # In eval mode, output should equal input
    assert np.allclose(output, x), "Dropout should not modify input in eval mode"

    # Backward should also pass through unchanged
    grad = np.ones_like(output)
    grad_input = dropout.backward(grad)
    assert np.allclose(grad_input, grad), "Gradient should be unchanged in eval mode"


def test_dropout_train_mode():
    """Test dropout in training mode (applies mask and scaling)."""

    p = 0.5
    dropout = nn.Dropout(p=p)
    dropout.train()

    x = np.ones((100, 100), dtype=np.float32) * 2.0
    output = dropout(x)

    # Check that approximately (1-p) fraction of elements are kept
    mask = output != 0.0
    kept_ratio = mask.sum() / mask.size
    expected_keep = 1.0 - p

    # Allow some tolerance for random variation
    assert (
        abs(kept_ratio - expected_keep) < 0.05
    ), f"Expected ~{expected_keep:.2f} kept, got {kept_ratio:.2f}"

    # Check that kept values are scaled correctly
    scaling_factor = 1.0 / (1.0 - p)
    assert np.allclose(
        output[mask], x[mask] * scaling_factor
    ), "Kept values should be scaled by 1/(1-p)"

    # Check backward pass
    grad = np.ones_like(output)
    grad_input = dropout.backward(grad)

    # Gradient should be masked and scaled the same way
    assert np.allclose(
        grad_input[mask], grad[mask] * scaling_factor
    ), "Gradient for kept values should be scaled by 1/(1-p)"
    assert np.all(
        grad_input[~mask] == 0.0
    ), "Gradient for dropped values should be zero"


def test_dropout_zero_probability():
    """Invalid probability values should raise ValueError"""

    with pytest.raises(ValueError):
        dropout = nn.Dropout(p=0.0)
