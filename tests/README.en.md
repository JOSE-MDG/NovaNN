# Testing in NovaNN v2.0.0

## 🧪 Overview

NovaNN includes a comprehensive suite of unit tests that verifies the correct implementation of all framework components. With >95% coverage, the tests ensure that every layer, optimizer, loss function, and utility works correctly in both forward and backward passes.

## 📁 Test Structure

```
📁 tests
├── 📁 initializers
│   └── 🐍 test_init.py
├── 📁 layers
│   ├── 📁 activations
│   │   ├── 🐍 test_leaky_relu.py
│   │   ├── 🐍 test_relu.py
│   │   ├── 🐍 test_sigmoid.py
│   │   ├── 🐍 test_softmax.py
│   │   └── 🐍 test_tanh.py
│   ├── 📁 batch_norm
│   │   ├── 🐍 test_batchnorm1d.py
│   │   └── 🐍 test_batchnorm2d.py
│   ├── 📁 conv
│   │   ├── 🐍 test_conv1d.py
│   │   └── 🐍 test_conv2d.py
│   ├── 📁 linear
│   │   └── 🐍 test_linear.py
│   ├── 📁 pooling
│   │   ├── 📁 gap
│   │   │   ├── 🐍 test_gap1d.py
│   │   │   └── 🐍 test_gap2d.py
│   │   └── 📁 maxpool
│   │       ├── 🐍 test_maxpooling1d.py
│   │       └── 🐍 test_maxpooling2d.py
│   └── 📁 regularization
│       └── 🐍 test_dropout.py
├── 📁 optimizers
│   ├── 🐍 test_adam.py
│   ├── 🐍 test_rmsprop.py
│   └── 🐍 test_sgd.py
└── 📁 sequential
    └── 🐍 test_sequential.py
```

### `📂 tests/`

**Unit test suite to verify the correct implementation of all NovaNN components**

Contains tests organized by modules that verify functionality, gradients, and behavior in different modes of all layers, optimizers, initializers, and framework utilities.

#### `📂 tests/📂 dataloader/`

##### `test_dataloader.py`

- **Purpose**: Verify the correct behavior of `DataLoader`, especially handling of the last batch when dataset size is not a multiple of batch size
- **Main tests**:
  - `test_last_batch_size()`: Ensures the last batch has the correct size (2 samples when batch_size=4 and dataset of 10 samples)
- **Methodology**:
  - Creates synthetic dataset of 10 samples with batch_size=4
  - Verifies that 3 batches are produced (4, 4, 2 samples)
  - Confirms the last batch has exactly 2 samples

#### `📂 tests/📂 initializers/`

##### `test_init.py`

- **Purpose**: Verify weight initialization functions (`kaiming_normal_`, `kaiming_uniform_`, `xavier_normal_`, `xavier_uniform_`, `random_init_`)
- **Main tests**:
  - `test_kaiming_normal_distribution()`: Verifies mean ≈0 and correct standard deviation for Kaiming normal
  - `test_kaiming_uniform_distribution()`: Verifies values are within calculated uniform limits
  - `test_xavier_normal_distribution()`: Verifies mean ≈0 and correct variance for Xavier normal
  - `test_xavier_uniform_distribution()`: Verifies uniform limits for Xavier uniform
  - `test_random_initializer()`: Verifies mean ≈0 for small random initialization
  - `test_exceptions_of_init_methods()`: Verifies that unsupported nonlinearities raise `ValueError`
- **Methodology**:
  - Tests with multiple tensor shapes (2D to 5D)
  - Compares sample statistics with expected theoretical values
  - Uses `calculate_gain` and `shape_validation` from `novann.core.init`
  - Empirical tolerances (0.1) for statistics

#### `📂 tests/📂 layers/📂 activations/`

**Tests to verify the correct implementation of activation functions**

##### `test_relu.py`

- **Purpose**: Verify the `ReLU` layer (Rectified Linear Unit)
- **Tests**:
  - `test_relu_forward_backward_and_numeric()`: Checks forward (non-negativity), backward (gradient mask), and numerical gradient for non-zero inputs
- **Methodology**:
  - Forward: Verifies shape and `max(0, x)` property
  - Backward: Compares with mask `(x > 0)`
  - Numerical gradient: Uses `numeric_grad_elementwise` to validate analytical gradients
  - Excludes `x = 0` where derivative is not defined

##### `test_leaky_relu.py`

- **Purpose**: Verify the `LeakyReLU` layer with configurable negative slope
- **Tests**:
  - `test_leaky_relu_forward_backward_and_numeric()`: Checks forward (piecewise behavior), backward (piecewise gradient), and numerical gradient
- **Methodology**:
  - Forward: Verifies `x` if `x ≥ 0`, `slope * x` if `x < 0`
  - Backward: Compares with `1` (x ≥ 0) and `slope` (x < 0)
  - Numerical gradient: Validation with finite differences for non-zero inputs

##### `test_sigmoid.py`

- **Purpose**: Verify the `Sigmoid` layer
- **Tests**:
  - `test_sigmoid_forward_backward_and_numeric()`: Checks forward (range (0,1)), backward (gradient), and numerical gradient
- **Methodology**:
  - Forward: Verifies shape and range `0 < σ(x) < 1`
  - Backward: Compares with analytical formula `σ(x) * (1 - σ(x))`
  - Numerical gradient: Complete validation with `numeric_grad_elementwise`

##### `test_softmax.py`

- **Purpose**: Verify the `Softmax` layer with numerical stability and probability properties
- **Tests**:
  - `test_softmax_forward_properties_and_shift_invariance_columnwise()`: Verifies forward properties (sums to 1 per row, non-negativity, shift invariance)
  - `test_softmax_backward_numeric_columnwise()`: Verifies backward using Jacobian-vector product and numerical gradient
- **Methodology**:
  - Forward: Sum to 1, non-negativity, invariance to additive constant
  - Backward: Compares analytical gradient with numerical approximation using `numeric_grad_scalar_from_softmax`

##### `test_tanh.py`

- **Purpose**: Verify the `Tanh` layer (hyperbolic tangent)
- **Tests**:
  - `test_tanh_forward_backward_and_numeric()`: Checks forward (range (-1,1), odd function property), backward (gradient), and numerical gradient
- **Methodology**:
  - Forward: Verifies shape, range `-1 < tanh(x) < 1` and property `tanh(-x) = -tanh(x)`
  - Backward: Compares with analytical formula `1 - tanh²(x)`
  - Numerical gradient: Validation with `numeric_grad_elementwise`

#### `📂 tests/📂 layers/📂 batch_norm/`

**Tests to verify Batch Normalization implementations in 1D and 2D**

##### `test_batchnorm1d.py`

- **Purpose**: Verify the `BatchNorm1d` layer for batch normalization in 1D/2D inputs
- **Tests**:
  - `test_batchnorm1d_forward_train_mode()`: Verifies forward in training mode (centering and feature normalization, moving statistics update)
  - `test_batchnorm1d_forward_eval_mode()`: Verifies forward in evaluation mode (use of moving statistics, no update)
  - `test_batchnorm1d_backward_gradient_check()`: Verifies analytical vs numerical gradients for parameters `gamma` and `beta`
  - `test_batchnorm1d_momentum_and_eps()`: Verifies custom momentum and epsilon parameters
  - `test_batchnorm1d_parameters()`: Verifies that the `parameters()` method returns the correct parameters
- **Methodology**:
  - Training mode: Verifies mean ≈0 and variance ≈1 per feature after normalization
  - Evaluation mode: Verifies use of moving statistics and numerical stability
  - Gradients: Uses `numeric_grad_wrt_param` to compare analytical and numerical gradients of `gamma` and `beta`
  - Parameters: Verifies shapes of moving statistics and parameter lists

##### `test_batchnorm2d.py`

- **Purpose**: Verify the `BatchNorm2d` layer for batch normalization in 2D convolutional inputs (4D)
- **Tests**:
  - `test_batchnorm2d_forward_train_mode()`: Verifies forward in training mode for 4D data (per-channel normalization over spatial dimensions)
  - `test_batchnorm2d_forward_eval_mode()`: Verifies forward in evaluation mode with moving statistics
  - `test_batchnorm2d_backward_gradient_check()`: Verifies gradients of `gamma` and `beta` with numerical gradients
  - `test_batchnorm2d_momentum_and_eps()`: Verifies momentum and epsilon parameters
  - `test_batchnorm2d_different_spatial_sizes()`: Verifies behavior with different spatial sizes
  - `test_batchnorm2d_parameters()`: Verifies `parameters()` method
- **Methodology**:
  - Training mode: Verifies mean ≈0 and variance ≈1 per channel (reduction over batch, height, width axes)
  - Evaluation mode: Verifies use of moving statistics without update
  - Gradients: Compares analytical gradients of `gamma` and `beta` with numerical approximations
  - Spatial sizes: Tests with different heights and widths, verifying shape preservation
  - Moving statistics: Verifies shapes `(1, C, 1, 1)` for broadcasting

#### `📂 tests/📂 layers/📂 conv/`

**Tests for 1D and 2D convolutional layers**

##### `test_conv1d.py`

- **Purpose**: Verify the `Conv1d` layer (1D convolution for sequence processing)
- **Main tests**:
  - `test_conv1d_forward_shape()`: Verifies output shape in forward pass with different configurations
  - `test_conv1d_forward_no_bias()`: Verifies forward without bias term
  - `test_conv1d_backward_gradient_check()`: Verifies gradients of weights and bias by comparison with numerical gradients
  - `test_conv1d_padding_modes()`: Tests different padding modes (zeros, reflect, replicate, circular)
  - `test_conv1d_parameters()`: Verifies that the `parameters()` method returns the correct parameters
- **Methodology**:
  - Uses deterministic RNG for reproducibility
  - Calculates expected shapes using formulas: $L_{out} = \lfloor\frac{L_{in} + 2 \times \text{padding} - K}{\text{stride}}\rfloor + 1$
  - For gradient verification: compares analytical gradients (`layer.weight.grad`, `layer.bias.grad`) with numerical approximations using `numeric_grad_wrt_param`
  - Tolerance `THRESHOLD=5e-3` for maximum differences

##### `test_conv2d.py`

- **Purpose**: Verify the `Conv2d` layer (2D convolution for image processing)
- **Main tests**:
  - `test_conv2d_forward_shape()`: Verifies output shapes in 4D
  - `test_conv2d_forward_no_bias()`: Verifies forward without bias
  - `test_conv2d_backward_gradient_check_small()`: Verifies gradients with small inputs for efficiency
  - `test_conv2d_different_kernel_stride_padding()`: Tests combinations of kernel, stride and padding (including tuples for separate dimensions)
  - `test_conv2d_padding_modes()`: Tests different padding modes
  - `test_conv2d_parameters()`: Verifies `parameters()` method
- **Methodology**:
  - Calculates expected dimensions: $H_{out} = \lfloor\frac{H_{in} + 2 \times p_h - K_h}{s_h}\rfloor + 1$, similar for width
  - Gradient verification with reduced inputs (`6x6`) to maintain manageable execution times
  - Same tolerance `THRESHOLD=5e-3` for comparisons
  - Support for asymmetric configurations (kernels, strides, paddings as tuples)

#### `📂 tests/📂 layers/📂 linear/`

**Tests for linear layers (fully connected)**

##### `test_linear.py`

- **Purpose**: Verify the `Linear` layer (fully connected linear transformation)
- **Main tests**:
  - `test_linear_forward_shape()`: Verifies output shape `(batch, out_features)`
  - `test_linear_forward_no_bias()`: Verifies forward without bias term
  - `test_linear_backward_gradient_check()`: Verifies gradients of weights and bias with numerical gradients
- **Methodology**:
  - Uses deterministic RNG
  - Verifies shapes and data types (`dtype=np.float32`)
  - Compares analytical vs numerical gradients using `numeric_grad_wrt_param` for both parameters (weight, bias)
  - Tolerance `THRESHOLD=5e-3`

#### `📂 tests/📂 layers/📂 pooling/`

**Tests for pooling layers (dimensional reduction)**

##### `tests/layers/pooling/gap/`

**Tests for Global Average Pooling**

##### `test_gap1d.py`

- **Purpose**: Verify the `GlobalAvgPool1d` layer (global average pooling in 1D)
- **Main tests**:
  - `test_global_avg_pool1d_forward_shape()`: Verifies collapse of length dimension to 1
  - `test_global_avg_pool1d_forward_values()`: Verifies correct average calculation with constant values
  - `test_global_avg_pool1d_backward_gradient()`: Verifies gradient with numerical comparison
  - `test_global_avg_pool1d_uniform_gradient()`: Verifies uniform gradient distribution (each element receives $1/L$)
- **Methodology**:
  - Forward: verifies shape `(batch, channels, 1)` and average values
  - Backward: uses `numeric_grad_scalar_wrt_x` for numerical comparison
  - Uniform distribution: verifies gradient is $1/L$ where $L$ is original length

##### `test_gap2d.py`

- **Purpose**: Verify the `GlobalAvgPool2d` layer (global average pooling in 2D)
- **Main tests**:
  - `test_global_avg_pool2d_forward_shape()`: Verifies collapse of spatial dimensions to `1x1`
  - `test_global_avg_pool2d_forward_values()`: Verifies average calculation with constant values
  - `test_global_avg_pool2d_backward_gradient()`: Verifies gradient with numerical comparison
  - `test_global_avg_pool2d_uniform_gradient()`: Verifies uniform gradient distribution (each element receives $1/(H \times W)$)
- **Methodology**:
  - Similar to `test_gap1d.py` but for 4D tensors
  - Verifies shape `(batch, channels, 1, 1)`
  - Uniform distribution over spatial area

#### `📂 tests/📂 layers/📂 pooling/📂 maxpool/`

**Tests for Max Pooling**

##### `test_maxpooling1d.py`

- **Purpose**: Verify the `MaxPool1d` layer (max pooling in 1D)
- **Main tests**:
  - `test_maxpool1d_forward_shape()`: Verifies output shape with kernel=2, stride=2
  - `test_maxpool1d_forward_padding()`: Verifies shape with padding
  - `test_maxpool1d_backward_gradient()`: Verifies gradient with numerical comparison
  - `test_maxpool1d_stride_different()`: Verifies with stride different from kernel
- **Methodology**:
  - Calculates expected dimensions using convolution formula
  - Backward: comparison with `numeric_grad_scalar_wrt_x`
  - Tolerance `THRESHOLD=5e-3`

##### `test_maxpooling2d.py`

- **Purpose**: Verify the `MaxPool2d` layer (max pooling in 2D)
- **Main tests**:
  - `test_maxpool2d_forward_shape()`: Verifies output shape with kernel=2, stride=2
  - `test_maxpool2d_forward_padding()`: Verifies shape with padding
  - `test_maxpool2d_backward_gradient()`: Verifies gradient with numerical comparison
- **Methodology**:
  - Similar to `test_maxpooling1d.py` but for 2D
  - Verifies 4D shapes
  - Same tolerance for gradient comparison

#### `📂 tests/📂 layers/📂 regularization/`

**Tests for regularization layers**

##### `test_dropout.py`

- **Purpose**: Verify the `Dropout` layer (regularization by random neuron dropout)
- **Main tests**:
  - `test_dropout_eval_mode()`: Verifies that in evaluation mode no dropout is applied (input passes unchanged)
  - `test_dropout_train_mode()`: Verifies that in training mode random mask and correct scaling are applied
  - `test_dropout_zero_probability()`: Verifies that invalid probabilities (p=0.0) raise `ValueError`
- **Methodology**:
  - Evaluation mode: Checks that input and output are identical, and gradients pass unchanged
  - Training mode: Verifies that approximately `(1-p)` fraction of elements are preserved, that preserved values scale by `1/(1-p)`, and that gradients are masked and scaled in the same way
  - Parameter validation: Checks that only probabilities in range `(0, 1)` are accepted
- **Details**:
  - Uses large test tensors (`100x100`) to obtain reliable statistics
  - 5% tolerance for random variation in proportion of preserved elements
  - Verifies consistency between forward and backward (same mask and scaling)

#### `📂 tests/📂 optimizers/`

**Tests for optimizers**

##### `test_adam.py`

- **Purpose**: Verify the `Adam` optimizer (Adaptive Moment Estimation)
- **Main tests**:
  - `test_adam_basic_update()`: Verifies that Adam updates parameters of a `Linear` layer
  - `test_adam_with_conv_layer()`: Verifies that Adam works with convolutional layers
  - `test_adam_bias_correction()`: Verifies bias correction mechanism in early steps
- **Methodology**:
  - Checks that parameters change after `step()`
  - Verifies that step counter (`t`) increments
  - For bias correction: runs multiple steps and verifies all updates are non-null
  - Uses real layers (`Linear`, `Conv2d`) with simulated forward/backward
- **Integration**: Depends on `Adam` from `novann/optim/` and framework layers

##### `test_rmsprop.py`

- **Purpose**: Verify the `RMSprop` optimizer (Root Mean Square Propagation)
- **Main tests**:
  - `test_rmsprop_basic_update()`: Verifies basic parameter update
  - `test_rmsprop_with_weight_decay()`: Verifies effect of weight decay (L2) on parameter magnitude
  - `test_rmsprop_zero_grad()`: Verifies that `zero_grad()` clears gradients
- **Methodology**:
  - Compares parameters before and after `step()` to confirm update
  - For weight decay: compares two identical models (with and without decay) after one optimization step
  - For `zero_grad()`: verifies all gradients are set to zero
- **Note**: The weight decay test currently verifies that norms are equal (with tolerance), which could be refined to verify that norm with decay is smaller.

##### `test_sgd.py`

- **Purpose**: Verify the `SGD` optimizer (Stochastic Gradient Descent) with momentum
- **Main tests**:
  - `test_sgd_basic_update()`: Verifies basic update in a `Sequential` model with multiple layers
  - `test_sgd_with_momentum()`: Verifies effect of momentum in consecutive updates
  - `test_sgd_zero_grad()`: Verifies that `zero_grad()` clears gradients
- **Methodology**:
  - Uses a `Sequential` model with two `Linear` layers for integral testing
  - For momentum: executes two steps with the same gradient and verifies the second step is non-null (velocity accumulation)
  - For `zero_grad()`: verifies gradients exist before and are zero after

#### `📂 tests/📂 sequential/`

**Tests for the Sequential container (layer stacking)**

##### `test_sequential.py`

- **Purpose**: Verify the `Sequential` container, which allows stacking multiple layers and executing them in sequence, both in forward and backward passes, including mode handling (train/eval) and initialization utilities.
- **Main tests**:
  - `test_sequential_linear_activation()`: Verifies sequences with linear layers and varied activation functions (ReLU, LeakyReLU, Sigmoid, Tanh), checking output shapes and expected ranges.
  - `test_sequential_conv_pooling()`: Verifies sequences with convolutional layers (Conv1d, Conv2d) and pooling layers (MaxPool, GlobalAvgPool) for 1D and 2D processing.
  - `test_sequential_mixed_layers()`: Verifies complex sequences with mixed layers (Conv, Dropout, Flatten, Linear, Softmax) and differentiated behavior in train vs eval modes.
  - `test_sequential_backward()`: Verifies complete backward propagation through multiple layers, checking gradient shapes and existence of gradients in all parameters.
  - `test_sequential_initialization_helpers()`: Verifies internal methods `_find_next_activation` and `_find_last_activation` used for intelligent weight initialization.
  - `test_sequential_parameters_and_zero_grad()`: Verifies that `parameters()` returns all parameters of contained layers and that `zero_grad()` correctly clears gradients.
- **Methodology**:
  - Creates `Sequential` models with varied architectures (MLP, CNN).
  - In forward: passes synthetic input tensors and verifies shapes, output ranges, and properties (e.g., sums to 1 with Softmax).
  - In backward: calculates gradients with respect to random outputs and verifies correct propagation through all layers.
  - Train/eval modes: alternates between modes and verifies specific behaviors (e.g., Dropout active only in train).
  - Parameters: counts and verifies access to all `weight` and `bias` of internal layers.
  - Initialization utilities: simulates search for activation functions adjacent to linear layers for proper initialization (Kaiming/Xavier).
[file content end]