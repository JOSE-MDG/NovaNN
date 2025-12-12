import numpy as np
import novann as nn
import pytest

RNG = np.random.RandomState(8)


def test_sequential_linear_activation():
    """Test Sequential with Linear and different activation functions."""

    # Test with ReLU
    seq_relu = nn.Sequential(
        nn.Linear(in_features=10, out_features=5, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=5, out_features=2, bias=True),
        nn.Sigmoid(),
    )
    seq_relu.train()

    x = RNG.randn(4, 10).astype(np.float32)
    output = seq_relu(x)
    assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"
    assert np.all(output >= 0) and np.all(
        output <= 1
    ), "Sigmoid output should be in [0, 1]"

    # Test with LeakyReLU
    seq_lrelu = nn.Sequential(
        nn.Linear(in_features=8, out_features=4),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(in_features=4, out_features=1),
        nn.Tanh(),
    )
    seq_lrelu.train()

    x = RNG.randn(3, 8).astype(np.float32)
    output = seq_lrelu(x)
    assert output.shape == (3, 1), f"Expected (3, 1), got {output.shape}"
    assert np.all(output >= -1) and np.all(
        output <= 1
    ), "Tanh output should be in [-1, 1]"


def test_sequential_conv_pooling():
    """Test Sequential with convolutional and pooling layers."""

    # 1D case
    seq_1d = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.GlobalAvgPool1d(),
    )
    seq_1d.train()

    x_1d = RNG.randn(2, 3, 16).astype(np.float32)
    output_1d = seq_1d(x_1d)
    assert output_1d.shape == (2, 8, 1), f"Expected (2, 8, 1), got {output_1d.shape}"

    # 2D case
    seq_2d = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.GlobalAvgPool2d(),
    )
    seq_2d.train()

    x_2d = RNG.randn(2, 3, 32, 32).astype(np.float32)
    output_2d = seq_2d(x_2d)
    assert output_2d.shape == (
        2,
        32,
        1,
        1,
    ), f"Expected (2, 32, 1, 1), got {output_2d.shape}"


def test_sequential_mixed_layers():
    """Test Sequential with mixed layer types including dropout and flatten."""

    seq = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(p=0.3),
        nn.Flatten(),
        nn.Linear(in_features=8 * 13 * 13, out_features=128),  # input was 28x28
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=128, out_features=10),
        nn.Softmax(),
    )

    # Test in training mode
    seq.train()
    x = RNG.randn(4, 1, 28, 28).astype(np.float32)
    output_train = seq(x)
    assert output_train.shape == (4, 10), f"Expected (4, 10), got {output_train.shape}"
    assert np.allclose(
        output_train.sum(axis=1), 1.0, rtol=1e-5
    ), "Softmax outputs should sum to 1"

    # Test in eval mode
    seq.eval()
    output_eval = seq(x)
    assert output_eval.shape == (4, 10), f"Expected (4, 10), got {output_eval.shape}"


def test_sequential_backward():
    """Test backward pass through Sequential with different layers."""

    seq = nn.Sequential(
        nn.Linear(in_features=20, out_features=10),
        nn.Tanh(),
        nn.Linear(in_features=10, out_features=5),
        nn.Sigmoid(),
    )
    seq.train()

    x = RNG.randn(3, 20).astype(np.float32) * 0.1
    output = seq(x)

    # Random gradient w.r.t output
    grad_output = RNG.randn(*output.shape).astype(np.float32) * 0.1

    # Backward pass
    grad_input = seq.backward(grad_output)

    # Check shapes
    assert (
        grad_input.shape == x.shape
    ), f"Input gradient shape {grad_input.shape} != input shape {x.shape}"

    # Check that parameters have gradients
    params = seq.parameters()
    assert len(params) == 4, f"Expected 4 parameters, got {len(params)}"

    for param in params:
        assert param.grad is not None, "Parameter should have gradient after backward"
        assert (
            param.grad.shape == param.data.shape
        ), f"Gradient shape {param.grad.shape} != data shape {param.data.shape}"


def test_sequential_initialization_helpers():
    """Test internal helper methods for initialization."""

    # Create a Sequential with known activation pattern
    seq = nn.Sequential(
        nn.Linear(in_features=10, out_features=5),
        nn.ReLU(),
        nn.Linear(in_features=5, out_features=3),
        nn.Tanh(),
        nn.Linear(in_features=3, out_features=1),
    )

    # Test _find_next_activation
    activation_key, param = seq._find_next_activation(0)
    assert activation_key == "relu", f"Expected 'relu', got {activation_key}"
    assert param is None, "ReLU has no parameters"

    activation_key, param = seq._find_next_activation(2)
    assert activation_key == "tanh", f"Expected 'tanh', got {activation_key}"
    assert param is None, "Tanh has no parameters"

    # Test _find_last_activation for final layer
    activation_key, param = seq._find_last_activation(4)
    assert activation_key == "tanh", f"Expected 'tanh', got {activation_key}"

    # Test for LeakyReLU with parameter
    seq_lrelu = nn.Sequential(
        nn.Linear(in_features=8, out_features=4), nn.LeakyReLU(negative_slope=0.1)
    )

    activation_key, param = seq_lrelu._find_next_activation(0)
    assert activation_key == "leakyrelu", f"Expected 'leakyrelu', got {activation_key}"
    assert param == 0.1, f"Expected slope 0.1, got {param}"


def test_sequential_parameters_and_zero_grad():
    """Test parameters() method and zero_grad()."""

    seq = nn.Sequential(
        nn.Linear(in_features=10, out_features=5, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=5, out_features=2, bias=False),
        nn.Sigmoid(),
    )

    params = seq.parameters()
    # First Linear: weight + bias = 2
    # Second Linear: weight only = 1
    assert len(params) == 3, f"Expected 3 parameters, got {len(params)}"

    # Set some gradients
    for param in params:
        param.grad = np.ones_like(param.data)

    # Verify gradients exist
    for param in params:
        assert param.grad is not None
        assert np.all(param.grad == 1.0)

    # Zero gradients
    seq.zero_grad()

    # Verify gradients are None (`set_to_none=True`)
    for param in params:
        assert param.grad is None
