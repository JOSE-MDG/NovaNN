# 🧪 Tests - NovaNN

[![Tests](https://img.shields.io/badge/tests-passing-success?style=flat-square)]()
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen?style=flat-square)]()
[![Pytest](https://img.shields.io/badge/framework-pytest-orange?style=flat-square)]()

Complete test suite to validate all NovaNN components, from autograd operations to neural network layers and optimizers.

## Test Structure

Tests are organized in two levels:

- **`tests/`** (root): High-level component tests (modules `nn`, optimizers, serialization, public API)
- **`nova/autograd/tests/`**: Specific tests for the autograd system (see [autograd README](../nova/autograd/README.md))

## Function Tests (`tests/functional/`)

### `test_activation.py`

Comprehensive tests of activation functions:

- **ReLU**: forward (non-negative), backward (gradient checking)
- **LeakyReLU**: forward with negative slope, backward
- **GELU**: forward, backward (relaxed tolerances due to numerical complexity)
- **PReLU**: forward with learnable parameters, backward
- **Tanh**: forward (range [-1, 1]), backward
- **Sigmoid**: forward (range [0, 1]), backward
- **Softmax**: forward (sums to 1), backward
- **LogSoftmax**: forward (exp sums to 1), backward

All tests validate:

- Correct shapes
- Expected value ranges
- Analytical vs numerical gradients with `grad_check_wrt_inputs`

### `test_loss.py`

Complete suite of loss functions with multiple test classes:

**MSELoss (Mean Squared Error):**

- Basic calculations with known values
- Reduction modes (`none`, `mean`, `sum`)
- Per-element weights
- Gradient validation
- Edge cases (multidimensional tensors, zero loss)

**L1Loss:**

- Basic calculations
- Reduction modes
- Gradient validation

**SmoothL1Loss (Huber):**

- Calculations with different betas
- Convergence to L1 for large errors
- Gradient validation

**BCELoss (Binary Cross Entropy):**

- Basic calculations
- Perfect predictions
- Numerical stability (no NaN/Inf)
- Gradients

**BCEWithLogitsLoss:**

- Equivalence with sigmoid + BCE
- Positive class weighting (`pos_weight`)
- Stability with extreme logits

**NLLLoss (Negative Log Likelihood):**

- Basic and per-batch calculations
- Class weights
- Reduction modes

**CrossEntropyLoss:**

- Equivalence with log_softmax + NLL
- Class weights
- Gradients
- Perfect predictions

**KLDivLoss (Kullback-Leibler Divergence):**

- Basic calculations (≥ 0)
- Identical distributions (≈ 0)
- `log_target` mode
- Reduction modes including `batchmean`

**Common tests:**

- Helper function `_reduce` with all modes
- Edge cases (empty tensors, single element, large batches)
- Parameterized tests for all reduction modes

## NN Module Tests (`tests/nn/`)

### `test_batchnorm.py`

Complete tests of BatchNormalization in all its variants:

**Forward pass:**

- BatchNorm1d with 2D input (N, C) and 3D input (N, C, L)
- BatchNorm2d with 4D input (N, C, H, W)
- BatchNorm3d with 5D input (N, C, D, H, W)
- Normalization validation (mean ≈ 0, std ≈ 1)

**Backward pass:**

- Gradients with cross entropy loss
- Gradient checking with finite differences
- Tests for 1D, 2D and 3D

**Running statistics:**

- Update during training with exponential momentum
- No update during eval
- Use of running stats in eval mode

**Affine parameters:**

- Application of weight and bias
- Behavior without affine parameters (`affine=False`)

**Dimension validation:**

- Rejection of incorrect dimensions
- Appropriate error messages

**Edge cases:**

- `track_running_stats=False`
- `momentum=None` (cumulative average)
- `reset_parameters()`

### `test_container.py`

Exhaustive tests of `Sequential`:

**Construction:**

- With module list
- With `OrderedDict`
- Empty Sequential

**Forward pass:**

- Correct chaining
- With Linear layers

**Indexing and slicing:**

- `__getitem__` by index and slice
- `__setitem__`
- `__delitem__` (element and slice)
- Index out of range

**Methods:**

- `append`, `insert`, `extend`, `pop`
- Negative indices
- Pop with slices

**Iteration:**

- `__iter__`, `__len__`

**Representation:**

- Simple and empty `__repr__`
- Repeated modules

**Arithmetic:**

- Addition (`+`, `+=`)
- Multiplication (`*`, `*=`, `__rmul__`)
- Type validation
- Non-positive multiplication

### `test_conv.py`

Tests of convolutional layers:

**Conv1d, Conv2d, Conv3d:**

- Correct output shapes
- Gradients (analytical vs numerical)
- No bias (`bias=False`)
- Stride and padding
- Asymmetric kernels (Conv2d)
- Preservation of temporal dimensions (Conv3d)

### `test_dropout.py`

Tests of dropout regularization:

**Dropout, Dropout2d, Dropout3d:**

- Shape preservation
- Deactivation in eval mode
- Application in train mode (some values to 0)
- Zero probability (no dropout)
- Full channel dropout (2d and 3d)

### `test_flatten.py`

Tests of Flatten:

- Default flatten (all except batch)
- Gradients
- Custom dimension range
- Flatten including batch
- Single dimension

### `test_layernorm.py`

Tests of Layer Normalization:

- Correct shape
- Normalization properties (mean ≈ 0, var ≈ 1)
- Gradients
- Without affine parameters
- Multidimensional normalization

### `test_lazy_variants.py`

Tests of lazy variants (automatic dimension inference):

**LazyBatchNorm (1d, 2d, 3d):**

- Inference of `num_features`
- Parameters not initialized before first forward
- Subsequent forwards

**LazyConv (1d, 2d, 3d):**

- Inference of `in_channels`
- Subsequent forwards

**LazyLinear:**

- Inference of `in_features`
- Multidimensional input

### `test_linear.py`

Tests of Linear layer:

- Forward shape
- Gradients
- No bias
- Multidimensional input

### `test_pooling.py`

Tests of pooling operations:

**MaxPool (1d, 2d, 3d):**

- Shapes
- Gradients
- Padding
- Dilation
- Non-square kernels
- Dimension preservation

**AvgPool (1d, 2d, 3d):**

- Shapes
- Gradients
- Padding

**GlobalAvgPool (1d, 2d, 3d):**

- Reduction to unit dimensions
- Gradients

## Optimizer Tests (`tests/optim/`)

### `test_sgd.py`

Tests of Stochastic Gradient Descent:

- Basic step (descent in gradient direction)
- Momentum accumulation
- Weight decay (L2)
- Convergence on quadratic function
- Multiple parameters
- Handling of None gradients

### `test_adam.py`

Tests of Adam optimizer:

- Basic step
- Bias correction in early steps
- Adaptive learning rate
- Convergence on quadratic
- Coupled weight decay

### `test_adamw.py`

Tests of AdamW:

- Basic step
- Decoupled weight decay
- Difference vs Adam with weight decay
- Convergence
- Bias correction

### `test_rmsprop.py`

Tests of RMSprop:

- Basic step
- Adaptation to gradient scale
- Centered mode
- Convergence
- Momentum

### `test_common.py`

Common tests for all optimizers:

- Parameter groups with different learning rates
- State persistence between steps
- `zero_grad()` utility

### `test_schedulers.py`

Exhaustive tests of learning rate schedulers:

**StepLR:**

- Decay at intervals
- Convergence
- Update of `last_epoch`
- Different gammas
- State dict (save/load)

**CosineAnnealingLR:**

- Cosine progression
- Convergence to `eta_min`
- State dict

**OneCycleLR:**

- Cycle progression
- Cycle momentum (inverse to LR)
- Warm-up phase
- Cool-down phase
- Compatibility with Adam (betas)

**Integrated tests:**

- All schedulers with all optimizers
- State persistence between schedulers

## Serialization Tests (`tests/serialization/`)

### `test_save.py`

Tests of the `save` function:

**Basic saving:**

- Saving to file path (string and Path)
- Saving to buffer (BytesIO)
- Automatic creation of parent directories

**Object types:**

- Modules (Sequential, Linear)
- Tensors
- State dicts
- Regular dictionaries
- Lists of tensors

**Pickle protocols:**

- Tests with different protocols (0-4, HIGHEST_PROTOCOL)

**Error handling:**

- Error saving None
- Error with invalid file type
- Error in read-only directory
- Error with non-serializable objects (lambdas)

### `test_load.py`

Tests of the `load` function:

**Basic loading:**

- Loading from file path
- Loading from buffer

**Security:**

- Error loading non-registered classes with `weights_only=True`
- Successful loading of non-registered classes with `weights_only=False`
- Successful loading of registered classes

**Error handling:**

- Error with non-existent file
- Error with corrupt file
- Error with empty file

**Roundtrip:**

- Save/load of models
- Save/load of state dicts

## Public API Tests (`tests/`)

### `test_api.py`

Massive suite validating the entire NovaNN public API:

**Tensor creation:**

- `tensor`, `zeros`, `ones`, `empty`, `full`, `eye`
- `arange`, `linspace`
- `zeros_like`, `ones_like`, `full_like`

**Random functions:**

- `rand`, `randn`, `randint`, `randperm`
- `uniform`, `normal`
- `manual_seed` (reproducibility)

**Mathematical functions:**

- `abs`, `sqrt`, `exp`, `log`, `pow`
- `floor`, `ceil`, `sign`, `clamp`

**Trigonometric functions:**

- `sin`, `cos`, `tan`, `tanh`
- `arcsin`, `arccos`, `arctan`, `sec`

**Reduction:**

- `sum`, `mean`, `var`, `std`
- `max`, `min`, `maximum`, `minimum`
- `argmax`, `argmin`, `argsort`

**Linear algebra:**

- `dot`, `det`, `inv`, `trace`, `norm`

**Shape manipulation:**

- `reshape`, `permute`, `flatten`
- `unsqueeze`, `split`, `tile`, `repeat_interleave`, `pad`

**Concatenation:**

- `cat`, `stack`

**Comparison and logic:**

- `allclose`, `all`, `any`, `where`
- `isnan`, `isinf`, `argwhere`, `unique`

**Utilities:**

- `one_hot`, `as_strided`

**Context managers:**

- `no_grad()`, `enable_grad()`, `is_grad_enabled()`

**Dtypes:**

- Availability of all dtypes
- Correct usage in tensor creation

**Metadata:**

- `__version__` exists

### `test_binding_system.py`

Tests of the dynamic binding system:

**YAML loading:**

- Successful file loading
- Correct structure
- Definition of common operations

**Function generators:**

- `make_forward_func`: binary and unary operations
- `make_reverse_func`: reverse operations (`__radd__`, etc.)
- `make_method`: regular methods (`.add()`, etc.)
- `make_inplace_func`: in-place operations with validations

**Bootstrapping:**

- Methods correctly bound to Tensor
- Dunder methods work (`+`, `-`, `*`)
- Reverse methods work (`5 + tensor`)
- Regular methods work (`.add()`, `.mul()`)
- In-place methods work (`.add_()`, `.mul_()`)
- Unary operations (`-tensor`, `abs(tensor)`)
- Raw args (indexing maintains types)

**Edge cases:**

- Methods not bound twice
- Chained operations
- Mix of scalars and tensors

### `test_conversion.py`

Type conversion tests with `ensure_tensor`:

**Tensor passthrough:**

- No changes when not needed
- Copy when dtype changes
- Copy when requires_grad changes

**Conversion from numpy:**

- Basic arrays
- With specified dtype
- With requires_grad

**Conversion from Python:**

- int, float, bool
- Lists and nested lists

**Edge cases:**

- 0-dimensional arrays
- Empty arrays
- Complex dtypes (error handling)

### `test_creation_and_casting.py`

Quick tests of creation and dtype casting:

- Dtype preservation in operations
- Math, trigonometry, reduction
- Indexing, linear algebra, concatenation

### `test_dataset.py`

Tests of the base `Dataset` class:

- Correct length
- Simple, slice, list, tensor indexing
- Abstract methods raise NotImplementedError

### `test_loader.py`

Tests of `DataLoader`:

- Basic iteration over batches
- Last batch with correct size
- Shuffle produces different order
- No shuffle produces same order
- Loader length
- Empty dataset
- Multiple epochs
- Integration with training loop

### `test_mnist_loader.py`

Exhaustive tests of the MNIST loader:

**TestMnistDataClass:**

- Initialization of `MnistData` with tensors
- `__len__` method returns correct size
- `__getitem__` method with simple index and slicing

**TestLoadMnistData:**

- Basic loading as tensors (type, shape, dtype verification)
- 4D format with `tensor4d=True` (N, 1, 28, 28)
- Normalization (mean ≈ 0, std ≈ 1)
- No normalization (values in range [0, 255])
- Valid label range [0, 9]
- Different dtypes (float16, float32, float64)
- Output as numpy arrays (`as_tensor=False`)
- Consistency between splits (train/test/val)
- No data leakage between splits
- `requires_grad=False` by default
- Iteration by batches
- Normalization with numpy arrays
- Label distribution (all 10 classes present)
- Dataset slicing (ranges, negative indices)
- Memory efficiency (no excessive copies)

**TestEdgeCases:**

- Handling of invalid paths
- Access to individual samples (1D for features, scalar for label)
- Access to individual 4D samples (shape (1, 28, 28))

**TestMnistMemoryUsage:**

- Memory usage in basic loading (<500MB)
- 4D vs 2D tensor overhead (ratio 0.8-1.5x)
- Impact of normalization on memory
- Tracking with decorator `@measure_memory`
- Comparison between dtypes (float64 ≈ 2x float32)
- Memory when accessing batches (large vs small)
- Memory cleanup after loading
- Numpy vs tensors (ratio <3.0x)

### `test_fashion_loader.py`

Complete tests of the Fashion-MNIST loader:

**TestFashionDataClass:**

- Initialization of `FashionData`
- `__len__` and `__getitem__` methods

**TestLoadFashionMnistData:**

- Basic loading as tensors with type validation
- 4D format (N, 1, 28, 28)
- Statistical normalization (mean ≈ 0, std ≈ 1)
- No normalization (range [0, 255])
- Valid labels [0, 9] (10 clothing classes)
- Different dtypes (float16/32/64)
- Numpy output
- 4D format with normalization combined
- Dimensional consistency between splits
- No data leakage
- `requires_grad=False` by default
- Iteration by batches
- Normalization with numpy
- Distribution of all 10 classes
- Advanced slicing
- Memory efficiency
- Difference vs regular MNIST (different content)
- Relative sizes of splits (train > test, val)

**TestEdgeCases:**

- Invalid paths (raises Exception)
- Access to individual sample (1D features, scalar label)
- Individual 4D sample (1, 28, 28)
- Empty slices (shape[0] == 0)

**TestFashionMemoryUsage:**

- Memory in basic loading (<850MB)
- 4D vs 2D overhead (ratio 0.5-1.5x)
- Impact of normalization
- Decorator `@measure_memory`
- Dtype comparison (float64 ≈ 2x float32)
- Batch access (large uses more memory)
- Cleanup after loading
- Numpy vs tensors (ratio <3.0x)
- Memory similarity with MNIST (ratio 0.5-3.5x)
- Overhead of loading 3 splits (<1000MB)
- Multiple loads without memory leaks (variance <30%)

### `test_hooks.py`

Tests of the hook system:

**HooksHandle:**

- Creation and removal
- Multiple removals (safe)

**Tensor hooks:**

- `register_hook` on backward
- Hook removal
- Multiple hooks on same tensor

**Optimizer hooks:**

- Pre-step hooks
- Post-step hooks
- Execution order
- Hook removal

### `test_clip_grad.py`

Tests of gradient clipping utilities:

**TestClipping:**

- **clip*grad_norm***: Normalizes gradients to maximum norm
  - Training for 20 epochs with Sequential model
  - Verification that all gradients are under `max_norm=1.0`
  - Integration with SGD optimizer
- **clip*grad_value***: Clips gradients by absolute value
  - Clipping to threshold=0.5
  - Validation that all gradients are in [-threshold, +threshold]
  - Integration with complete training loop

Both tests verify:

- Correct clipping on all model parameters
- Compatibility with optimizers
- Does not break backward-optimizer flow

### `test_metrics.py`

Tests of evaluation metrics:

**Regression:**

- **MSE/RMSE**: basic, with error, reset, multiple batches
- **MAE**: basic, with error, robustness to outliers
- **R²Score**: perfect fit (=1), baseline (≈0), good fit

**Classification:**

- **Accuracy**: perfect, partial, zero
- **Precision/Recall/F1**: perfect, with errors
- **ConfusionMatrix**: binary, multiclass
- **ROCAUC**: perfect (=1), random (≈0.5), good separation

### `test_registry.py`

Tests of the registry system:

**`registry_class`:**

- Registration of simple classes
- Returns original class
- Idempotent registration
- `get_registered_classes` works
- Non-registered class returns None

### `test_memory_utils.py`

Tests of the memory monitoring system:

**TestMemoryTracker:**

- Basic context manager
- Memory properties (peak_mb, current_mb, peak_kb, current_kb)

**TestQuickMemoryCheck:**

- Profiling of simple functions
- Support for kwargs
- Returns results and statistics

**TestCompareMemory:**

- Comparison between two functions
- Memory usage ratio
- Validation that large function uses more memory

**TestMemoryTrackerAdvanced:**

- `get_top_stats(n)` returns top allocations
- Verbose mode prints statistics

**TestMemoryContextBehavior:**

- Sequential uses of tracker
- Consistency between multiple runs

### `test_timing_decorators.py`

Tests of benchmarking and timing tools:

**TestBenchmark:**

- Basic function benchmarking
- Support for kwargs
- Effect of warmup iterations (do not affect timing)
- Result consistency (low std for stable functions)
- Returns: result, mean_time, std_time

**TestChronometer:**

- Basic decorator `@chronometer`
- Multiple iterations with warmup
- Flag `return_time=True` returns (result, elapsed)
- Mode `verbose=False` suppresses output
- Warmup iterations validation
- Combination of return_time + n_iters returns average time

**`registry_op`:**

- Registration of Functions
- Returns original class
- Error with non-Function classes
- Idempotent registration
- Accessible operations

**Integration:**

- Both decorators together

## Testing Strategy

### Gradient Checking

Main validation method: **finite difference gradient checking**

```python
analytic, numeric = grad_check_wrt_inputs(operation, input, eps=1e-4)
assert nova.allclose(analytic[0], numeric[0], rtol=1e-2, atol=1e-3)
```

Compares:

- **Analytical gradients**: computed by `backward()`
- **Numerical gradients**: approximated by finite differences

### Tolerances

Adapted according to numerical stability:

- Stable operations: `rtol=1e-3, atol=5e-3`
- Complex operations: `rtol=1e-2, atol=5e-2`
- Very sensitive operations: `rtol=0.1, atol=0.1`

### Fixtures and Parameterization

```python
@pytest.mark.parametrize("optimizer", [SGD, Adam, AdamW, RMSprop])
def test_with_all_optimizers(optimizer):
    # Test runs with each optimizer
```

### Reproducibility

Fixed seed in all tests:

```python
nova.manual_seed(8)  # or nova.manual_seed(42)
```

## Running Tests

```bash
# All tests
poetry run pytest

# Verbose tests
poetry run pytest tests/ -v

# Tests with coverage
poetry run pytest --cov

# Tests with html report
poetry run pytest --cov --cov-report=html
```

## Test Coverage

**Current coverage: 87%**

Tests cover:

- ✅ Forward pass (shapes, values, properties)
- ✅ Backward pass (analytical vs numerical gradients)
- ✅ Edge cases (incorrect dimensions, extreme values)
- ✅ Special configurations (no bias, lazy initialization)
- ✅ Operation modes (train vs eval)
- ✅ State persistence (state_dict, load_state_dict)
- ✅ Component compatibility
- ✅ Complete public API
- ✅ Dynamic binding system
- ✅ Safe serialization
- ✅ Evaluation metrics
- ✅ Data loading

---

> For specific autograd tests (operations, engine, Function), see [`nova/autograd/tests/`](../nova/autograd/tests/) and the [autograd README](../nova/autograd/README.md).
