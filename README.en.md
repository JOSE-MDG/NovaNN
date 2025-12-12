![Banner](./images/NovaNN%20Banners.png)

![version](https://img.shields.io/badge/version-3.0.0-blue)
![python](https://img.shields.io/badge/python-v3.14-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![tests](https://img.shields.io/badge/tests-pytest-orange)
![coverage](https://img.shields.io/badge/coverage-95%25-success)

## 🌐 Available Languages

- 🇬🇧 [English](README.en.md)
- 🇪🇸 [Español](README.md)

**NovaNN** is a framework **that** provides tools and examples for creating **Fully Connected** and **convolutional** neural networks alongside modules that offer support and enhance network training. This project **demonstrates** a deep understanding and mastery of how these networks function, inspired by how the most popular deep learning frameworks like **PyTorch** and **TensorFlow** operate, with **PyTorch** being the primary inspiration for this project.

**Clarification**: This framework was created for educational purposes to gain a clear understanding of what major Deep Learning frameworks do. **Objective**: To demonstrate solid knowledge in: **neural networks**, **Deep Learning**, **Machine Learning**, **mathematics**, **software engineering**, **System Design**, **best practices**, **unit testing**, **ultra-modular design**, and **data preprocessing**.

## Introduction

 **NovaNN** features a completely **modular structure designed** to resemble a framework as closely as possible.

 The `data/` directory is intended for datasets such as _Fashion-MNIST_ and _MNIST_. Since the original files are not included in the repository due to their size, you can download them from **Kaggle** via the following links:
  - [fasion-mnist-train](https://www.kaggle.com/datasets/zalando-research/fashionmnist?select=fashion-mnist_train.csv)
  - [fasion-mnist-test](https://www.kaggle.com/datasets/zalando-research/fashionmnist?select=fashion-mnist_test.csv)
  - [mnist-train](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv)
  - [mnist-test](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv)

 The `examples/` directory contains example scripts such as **binary classification**, **multiclass classification**, **regression**, and **convolutional layers**.

 In `notebooks/` you will find a Jupyter notebook that prepares validation data from the downloaded datasets.
 **Important note**: Verify the data structure before running the notebook, as variations in the format may cause errors.

 It is also **necessary to create a `.env` file** with the following environment variables:

 - **FASHION_TRAIN_DATA_PATH**: Training data path
 - **EXPORTATION_FASHION_TRAIN_DATA_PATH**: Training data path separated from validation data.
 - **FASHION_VALIDATION_DATA_PATH**: Validation data path separated from training data.
 - **FASHION_TEST_DATA_PATH**: Test data path

 - **MNIST_TRAIN_DATA_PATH**: Training data path
 - **EXPORTATION_MNIST_TRAIN_DATA_PATH**: Training data path separated from validation data.
 - **MNIST_VALIDATION_DATA_PATH**: Validation data path separated from training data.
 - **MNIST_TEST_DATA_PATH**: Test data path

 - **LOG_FILE**: Log file path
 - **LOGGER_DEFAULT_FORMAT**: `%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s` <- Default value.
 - **LOGGER_DATE_FORMAT**: `%Y-%m-%d %H:%M:%S` <- Default value.

 - **Comparison with PyTorch**: The performance of **NovaNN** was evaluated against the **PyTorch** framework on a classification task with the _MNIST_ dataset, using the same dataset and hyperparameters in both implementations. The comparison results were saved in `json` format with metrics such as accuracy and loss.

 - **[main.py](main.py)**: This file implements the training code and the network structure to be used for the comparison.
 - **[colab](https://colab.research.google.com/drive/1M6Qo2vu4mjVJWQGBK6I4PFBvwwXbQvvj?usp=sharing)**: The notebook contains the PyTorch version of the training code, which performs the same procedure as the script.

 ### Comparison Results:

 Once the results were obtained, a script ([visualization](./novann/utils/visualizations/visualization.py)) was created to plot the results in a more presentable way.

 ![image](./images/metrics.png)

 ## 📂 Project Structure

 [NovaNN Structure](./NovaNNFiletree.md)

```
📁 NovaNN
├── 📁 data
│   ├── 📁 FashionMnist
│   │   └── .gitkeep
│   └── 📁 Mnist
│       └── .gitkeep
├── 📁 examples
│   ├── 🐍 binary_classification.py
│   ├── 🐍 conv_example.py
│   ├── 🐍 multiclass_classification.py
│   └── 🐍 regresion.py
├── 📁 images
│   └── 🖼️ metrics.png
├── 📁 notebooks
│   └── 📄 exploration.ipynb
├── 📁 novann
│   ├── 📁 _typing
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 _typing.py
│   ├── 📁 core
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 config.py
│   │   ├── 🐍 constants.py
│   │   └── 🐍 init.py
│   ├── 📁 layers
│   │   ├── 📁 activations
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 activations.py
│   │   │   ├── 🐍 relu.py
│   │   │   ├── 🐍 sigmoid.py
│   │   │   ├── 🐍 softmax.py
│   │   │   └── 🐍 tanh.py
│   │   ├── 📁 bn
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 batchnorm1d.py
│   │   │   └── 🐍 batchnorm2d.py
│   │   ├── 📁 convolutional
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 conv1d.py
│   │   │   └── 🐍 conv2d.py
│   │   ├── 📁 flatten
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 flatten.py
│   │   ├── 📁 linear
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 linear.py
│   │   ├── 📁 pooling
│   │   │   ├── 📁 gap
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 global_avg_pool1d.py
│   │   │   │   └── 🐍 global_avg_pool2d.py
│   │   │   ├── 📁 maxpool
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 maxpool1d.py
│   │   │   │   └── 🐍 maxpool2d.py
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 regularization
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 dropout.py
│   │   └── 🐍 __init__.py
│   ├── 📁 losses
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 functional.py
│   ├── 📁 metrics
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 metrics.py
│   ├── 📁 model
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 nn.py
│   ├── 📁 module
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 layer.py
│   │   └── 🐍 module.py
│   ├── 📁 optim
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 adam.py
│   │   ├── 🐍 rmsprop.py
│   │   └── 🐍 sgd.py
│   ├── 📁 utils
│   │   ├── 📁 decorators
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 timing.py
│   │   ├── 📁 gradient_checking
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 numerical.py
│   │   ├── 📁 log_config
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 logger.py
│   │   ├── 📁 train
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 train.py
│   │   ├── 📁 visualizations
│   │   │   └── 🐍 visualization.py
│   │   └── 🐍 __init__.py
│   └── 🐍 __init__.py
├── 📁 tests
│   ├── 📁 initializers
│   │   └── 🐍 test_init.py
│   ├── 📁 layers
│   │   ├── 📁 activations
│   │   │   ├── 🐍 test_leaky_relu.py
│   │   │   ├── 🐍 test_relu.py
│   │   │   ├── 🐍 test_sigmoid.py
│   │   │   ├── 🐍 test_softmax.py
│   │   │   └── 🐍 test_tanh.py
│   │   ├── 📁 batch_norm
│   │   │   ├── 🐍 test_batchnorm1d.py
│   │   │   └── 🐍 test_batchnorm2d.py
│   │   ├── 📁 conv
│   │   │   ├── 🐍 test_conv1d.py
│   │   │   └── 🐍 test_conv2d.py
│   │   ├── 📁 linear
│   │   │   └── 🐍 test_linear.py
│   │   ├── 📁 pooling
│   │   │   ├── 📁 gap
│   │   │   │   ├── 🐍 test_gap1d.py
│   │   │   │   └── 🐍 test_gap2d.py
│   │   │   └── 📁 maxpool
│   │   │       ├── 🐍 test_maxpooling1d.py
│   │   │       └── 🐍 test_maxpooling2d.py
│   │   └── 📁 regularization
│   │       └── 🐍 test_dropout.py
│   ├── 📁 optimizers
│   │   ├── 🐍 test_adam.py
│   │   ├── 🐍 test_rmsprop.py
│   │   └── 🐍 test_sgd.py
│   ├── 📁 sequential
│   │   └── 🐍 test_sequential.py
│   ├── 📝 README.en.md
│   └── 📝 README.md
├── ⚙️ .gitignore
├── 📄 LICENCE
├── 📝 NovaNNFiletree.md
├── 📝 README.en.md
├── 📝 README.md
├── 🐍 main.py
├── 📄 poetry.lock
└── ⚙️ pyproject.toml
```

## Module `novann/` and Subdirectories Structure

Here we will explain in detail what each submodule does and its components.

### `📂 _typing/`

**Type definitions for static type checking system**

Contains:
- `_typing.py`: Custom types for tensors, initializers, parameters, etc.

#### `_typing.py`

- **Purpose**: Type definitions (type hints) for the entire framework
- **Main types**:
  - `Shape`: Tensor shape (tuple of integers)
  - `InitFn`: Weight initialization function signature
  - `ListOfParameters`: List of trainable parameters
  - `IntOrPair`: Integer or tuple for flexible dimensions
  - `KernelSize`, `Stride`, `Padding`: Types for convolutional layers
  - `Optimizer`: Alias for optimizers (Adam, SGD, RMSprop)
  - `LossFunc`: Alias for loss functions
  - `Loader`: Type for iterable dataloaders
- **Usage in the framework**: These types are imported by all modules for consistent type annotations
- **Connections**:
  - `InitFn` is used by `config.py` for initialization maps
  - `ListOfParameters` is used by layers that return trainable parameters
  - Convolutional types are used by `Conv1d`, `Conv2d`, `MaxPool`, etc.

### `📂 core/`

**Global configuration, weight initialization and framework constants**

Contains:
- `config.py`: Weight initialization maps based on activations
- `init.py`: Initialization functions (Xavier, Kaiming, random)
- `constants.py`: Environment variables and dataset paths

#### `config.py`

- **Purpose**: Centralized weight initialization configuration for the framework
- **Initialization dictionaries**:
  - `DEFAULT_NORMAL_INIT_MAP`: Mapping of normal distribution initialization functions for different activation functions
  - `DEFAULT_UNIFORM_INIT_MAP`: Mapping of uniform distribution initialization functions for different activation functions
- **Supported keys**: `relu`, `leakyrelu`, `tanh`, `sigmoid`, `default` (for default initialization)
- **Integration with `core/init.py`**: Uses initialization functions (`kaiming_normal_`, `kaiming_uniform_`, `xavier_normal_`, `xavier_uniform_`, `random_init_`) and `calculate_gain` to compute appropriate gains
- **Usage in linear layers**: The maps are used by `Linear.reset_parameters()` to initialize weights and biases based on the adjacent activation
- **Usage in convolutional layers**: Also used by `Conv1d` and `Conv2d` to initialize convolutional kernels following the same principle
- **Usage in Sequential**: The `Sequential` container uses these maps for automatic initialization of linear and convolutional layers based on surrounding activations
- **Details per activation**:
  - **ReLU**: Kaiming normal/uniform initialization with `a=0.0`
  - **LeakyReLU**: Kaiming normal/uniform initialization with `a=0.01` (negative slope)
  - **Tanh**: Xavier normal/uniform initialization with gain calculated for tanh
  - **Sigmoid**: Xavier normal/uniform initialization with gain calculated for sigmoid
  - **Default**: Small random initialization (conservative) for unspecified cases

#### `init.py`

- **Purpose**: Weight initialization functions (Xavier/Glorot, Kaiming/He and random)
- **Usage in layers**: Used by linear and convolutional layers based on the following activation
- **Functions**:
  - `calculate_gain(nonlinearity, param)`: Calculates gain for activations
  - `xavier_normal_(shape, gain)`: Xavier normal initialization
  - `xavier_uniform_(shape, gain)`: Xavier uniform initialization
  - `kaiming_normal_(shape, a, nonlinearity, mode)`: Kaiming normal initialization
  - `kaiming_uniform_(shape, a, nonlinearity, mode)`: Kaiming uniform initialization
  - `random_init_(shape, gain)`: Small random initialization (conservative default)
- **Integration with `config.py`**: These functions are mapped by `DEFAULT_NORMAL_INIT_MAP` and `DEFAULT_UNIFORM_INIT_MAP`

#### `constants.py`

- **Purpose**: Global configuration variables from `.env` file
- **Content**:
  - Fashion-MNIST and MNIST dataset paths
  - Logging configuration (file, format, level)
- **Usage in the framework**: Imported by other modules for configuration access

### `📂 layers/`

**Implementations of all neural network layers (linear, convolutional, pooling, normalization, activations and regularization)**

Contains subdirectories organized by layer type:
- `activations/`: Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- `linear/`: Linear layers (fully connected)
- `convolutional/`: 1D and 2D convolutional layers
- `pooling/`: Pooling layers (MaxPool, GlobalAvgPool)
- `bn/`: Batch normalization (BatchNorm)
- `regularization/`: Regularization (Dropout)
- `flatten/`: Layer for flattening tensors

All layers inherit from `Layer` and follow the standard forward/backward interface.

#### `📂 layers/📂 activations/`

**Base classes and implementations of activation functions**

Contains:
- `activations.py`: Base class `Activation` for all activations
- `relu.py`: Implementations of `ReLU` and `LeakyReLU`
- `sigmoid.py`: Implementation of `Sigmoid`
- `softmax.py`: Implementation of `Softmax`
- `tanh.py`: Implementation of `Tanh`

##### `activations.py`

- **Purpose**: Base class for all activation layers
- **Main class**: `Activation` (inherits from `Layer`)
- **Attributes**:
  - `name`: Lowercase class name (identifier for initialization maps)
  - `affect_init`: Boolean indicating if the activation influences weight initialization
- **Methods**:
  - `get_init_key()`: Returns the initialization key if `affect_init = True`
  - `init_key`: Property that is an alias for `get_init_key()`
- **Connections**:
  - Inherits from `Layer` (`novann.module.layer`)
  - Used by all concrete activations (ReLU, Sigmoid, etc.)
  - The `affect_init` attribute and `get_init_key()` are used by `Sequential` for automatic initialization based on activations

##### `relu.py`

- **Purpose**: Implementations of ReLU and LeakyReLU
- **Classes**:
  - `ReLU`: Rectified Linear Unit (max(0, x))
  - `LeakyReLU`: Leaky ReLU with configurable negative slope
- **Attributes**:
  - `ReLU._mask`: Boolean mask saved during forward (x > 0)
  - `LeakyReLU.a`: Negative slope for negative values
  - `LeakyReLU.activation_param`: Stores the same value as `a` (for consistency)
  - `LeakyReLU._cache_input`: Input saved for backward
- **Connections**:
  - Both classes inherit from `Activation`
  - `affect_init = True` in both, so they influence weight initialization
  - Use `kaiming_normal_`/`kaiming_uniform_` from `config.py` for initialization (mapped by `relu` and `leakyrelu`)
- **Implementation**:
  - `ReLU.forward`: Applies `max(0, x)` and saves mask
  - `ReLU.backward`: Propagates gradients only where input was > 0
  - `LeakyReLU.forward`: Applies `x if x >= 0, else a * x`
  - `LeakyReLU.backward`: Gradient is `1.0` for x ≥ 0, `a` for x < 0

##### `sigmoid.py`

- **Purpose**: Implementation of the sigmoid function
- **Class**: `Sigmoid` (inherits from `Activation`)
- **Attributes**:
  - `out`: Saved output from forward for use in backward
  - `affect_init = True`: Affects weight initialization
- **Connections**:
  - Uses `xavier_normal_`/`xavier_uniform_` from `config.py` for initialization (mapped by `sigmoid`)
  - Uses gain calculated by `calculate_gain` from `init.py`
- **Implementation**:
  - `forward`: Calculates `1 / (1 + exp(-x))` and saves in `out`
  - `backward`: Calculates gradient using `out * (1 - out)`

##### `softmax.py`

- **Purpose**: Implementation of numerically stable softmax
- **Class**: `Softmax` (inherits from `Activation`)
- **Attributes**:
  - `axis`: Axis over which to apply softmax (default 1)
  - `out`: Saved output from forward
  - `affect_init = False`: Does not affect weight initialization (designed to be used with `CrossEntropyLoss`)
- **Connections**:
  - Normally used with `CrossEntropyLoss` which combines softmax and loss
  - Has no entry in the initialization maps of `config.py`
- **Implementation**:
  - `forward`: Numerically stable softmax (subtract maximum before exponentiating)
  - `backward`: Calculates efficient Jacobian-vector product using cached output

##### `tanh.py`

- **Purpose**: Implementation of hyperbolic tangent
- **Class**: `Tanh` (inherits from `Activation`)
- **Attributes**:
  - `out`: Saved output from forward for use in backward
  - `affect_init = True`: Affects weight initialization
- **Connections**:
  - Uses `xavier_normal_`/`xavier_uniform_` from `config.py` for initialization (mapped by `tanh`)
  - Uses gain calculated by `calculate_gain` from `init.py`
- **Implementation**:
  - `forward`: Calculates `tanh(x)` and saves in `out`
  - `backward`: Calculates gradient using `1 - tanh(x)^2`

#### `📂 layers/📂 bn/`

**Batch Normalization implementations for different input dimensions**

Contains:
- `batchnorm1d.py`: Batch Normalization for 1D/2D inputs (fully connected and 1D convolutions)
- `batchnorm2d.py`: Batch Normalization for 2D convolutional inputs (4D tensors)

##### `batchnorm1d.py`

- **Purpose**: Batch Normalization implementation for 1D/2D inputs, compatible with fully connected layers and 1D convolutions
- **Main class**: `BatchNorm1d`
- **Main features**:
  - Support for 2D inputs `(batch, features)` and 3D inputs `(batch, channels, sequence_length)`
  - Different modes for training (batch statistics) and evaluation (moving statistics)
  - Learnable parameters `gamma` (scale) and `beta` (shift) with default initialization (1s and 0s)
  - Numerical stability term `eps` and Bessel's correction for unbiased variance
  - Configurable momentum for moving statistics update
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - Uses `Parameters` from `novann.module` for trainable parameters `gamma` and `beta`
  - Uses `ListOfParameters` from `novann._typing` for parameter return
  - Parameters `gamma` and `beta` are automatically excluded from weight decay in optimizers
- **Usage in the framework**:
  - After `Linear` layers in fully connected networks to stabilize activations
  - After `Conv1d` layers in 1D convolutional networks for per-channel normalization
- **Technical details**:
  - **Algorithm (Training Mode)**:

    **Batch statistics calculation**:
    
    $$\mu = \frac{1}{m} \sum_{i=1}^{m} x_i$$
    
    $$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2$$

    **Bessel's correction (unbiased variance)**:
    
    $$\sigma_{\text{unbiased}}^2 = \sigma^2 \cdot \frac{m}{m - 1} \quad \text{(if } m > 1\text{)}$$

    **Normalization**:
    
    $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma_{\text{unbiased}}^2 + \epsilon}}$$

    **Scale and shift**:
    
    $$y_i = \gamma \hat{x}_i + \beta$$

    **Moving statistics update**:
    
    $$\text{running\_mean} = (1 - \text{momentum}) \cdot \text{running\_mean} + \text{momentum} \cdot \mu$$
    
    $$\text{running\_var} = (1 - \text{momentum}) \cdot \text{running\_var} + \text{momentum} \cdot \sigma_{\text{unbiased}}^2$$

  - **Algorithm (Evaluation Mode)**:

    **Normalization with moving statistics**:
    
    $$\hat{x}_i = \frac{x_i - \text{running\_mean}}{\sqrt{\text{running\_var} + \epsilon}}$$
    
    $$y_i = \gamma \hat{x}_i + \beta$$

  - **Backward Pass**:

    **Parameter gradients**:
    
    $$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$
    
    $$\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}$$

    **Gradient with respect to input** (efficient vectorized version):
    
    $$\frac{\partial L}{\partial x_i} = \frac{\gamma}{m \sqrt{\sigma^2 + \epsilon}} \left( m \cdot \frac{\partial L}{\partial \hat{x}_i} - \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j \right)$$

  - **Dimension handling**:
    - **2D inputs**: Reduction axes = `(0,)` (batch)
    - **3D inputs**: Reduction axes = `(0, 2)` (batch and sequence)

##### `batchnorm2d.py`

- **Purpose**: Batch Normalization implementation for 2D convolutional inputs (4D tensors)
- **Main class**: `BatchNorm2d`
- **Main features**:
  - Specifically designed for 4D tensors `(batch, channels, height, width)`
  - Per-channel normalization over spatial and batch dimensions
  - Parameters `gamma` and `beta` with shape `[1, channels, 1, 1]` for broadcasting
  - Training/evaluation modes with differentiated behavior
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - Uses `Parameters` for `gamma` and `beta` with shape adapted to 4D tensors
  - Compatible with `Conv2d` layers for 2D convolutional networks
- **Usage in the framework**:
  - After `Conv2d` layers in 2D convolutional networks
  - Normalizes activations per channel before activation functions
- **Technical details**:
  - **Algorithm (Training Mode)**:

    **Per-channel statistics calculation**:
    
    $$\mu_c = \frac{1}{m \cdot H \cdot W} \sum_{n=1}^{m} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{nchw}$$
    
    $$\sigma_c^2 = \frac{1}{m \cdot H \cdot W} \sum_{n=1}^{m} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{nchw} - \mu_c)^2$$

    **Per-channel normalization**:
    
    $$\hat{x}_{nchw} = \frac{x_{nchw} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$
    
    $$y_{nchw} = \gamma_c \hat{x}_{nchw} + \beta_c$$

  - **Algorithm (Evaluation Mode)**:

    **Normalization with per-channel moving statistics**:
    
    $$\hat{x}_{nchw} = \frac{x_{nchw} - \text{running\_mean}_c}{\sqrt{\text{running\_var}_c + \epsilon}}$$
    
    $$y_{nchw} = \gamma_c \hat{x}_{nchw} + \beta_c$$

  - **Backward Pass**:

    **Per-channel parameter gradients**:
    
    $$\frac{\partial L}{\partial \gamma_c} = \sum_{n=1}^{m} \sum_{h=1}^{H} \sum_{w=1}^{W} \frac{\partial L}{\partial y_{nchw}} \cdot \hat{x}_{nchw}$$
    
    $$\frac{\partial L}{\partial \beta_c} = \sum_{n=1}^{m} \sum_{h=1}^{H} \sum_{w=1}^{W} \frac{\partial L}{\partial y_{nchw}}$$

    **Gradient with respect to input** (similar to `BatchNorm1d` but reducing over `(0, 2, 3)`):
    
    $$\frac{\partial L}{\partial x_{nchw}} = \frac{\gamma_c}{m \cdot H \cdot W \cdot \sqrt{\sigma_c^2 + \epsilon}} \left( m \cdot H \cdot W \cdot \frac{\partial L}{\partial \hat{x}_{nchw}} - \sum_{n',h',w'} \frac{\partial L}{\partial \hat{x}_{n'ch'w'}} - \hat{x}_{nchw} \sum_{n',h',w'} \frac{\partial L}{\partial \hat{x}_{n'ch'w'}} \hat{x}_{n'ch'w'} \right)$$

#### `📂 layers/📂 convolutional/`

**Implementations of convolutional layers for 1D and 2D signal processing**

Contains:
- `conv1d.py`: 1D convolutional layer for sequence and temporal signal processing
- `conv2d.py`: 2D convolutional layer for image and spatial data processing

##### `conv1d.py`

- **Purpose**: Implements a 1D convolutional layer for processing sequences and temporal signals
- **Main class**: `Conv1d`
- **Main features**:
  - Support for 3D inputs `(batch_size, channels, sequence_length)`
  - 1D convolutional kernel with configurable size
  - Configurable stride and padding along the temporal dimension
  - Multiple padding modes same as `Conv2d`
  - Efficient implementation via `im2col` similar to `Conv2d`
  - Initialization with `DEFAULT_UNIFORM_INIT_MAP["relu"]` by default
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - Uses `Parameters` for trainable weights and biases
  - Uses custom types from `novann._typing`
  - Compatible with `BatchNorm1d` for batch normalization in sequences
  - Can be used with 1D pooling layers (`MaxPool1d`, `GlobalAvgPool1d`)
- **Usage in the framework**:
  - For processing temporal sequences (audio, time series)
  - As a component in 1D convolutional networks
- **Technical details**:

  **im2col transformation for 1D**:
  
  $$\text{col} = \text{im2col}(x) \quad \text{(shape: } C_{in} \times K \text{, } N \times L_{out})$$
  
  $$W_{col} = \text{reshape}(W) \quad \text{(shape: } C_{out} \text{, } C_{in} \times K)$$
  
  $$\text{out} = W_{col} \times \text{col} \quad \text{(shape: } C_{out} \text{, } N \times L_{out})$$

  **Output length calculation**:
  
  $$L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - K}{\text{stride}}\right\rfloor + 1$$

  **Gradients** (similar to `Conv2d` but in 1D):
  
  $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \text{out}} \times \text{col}^T$$
  
  $$\frac{\partial L}{\partial \text{bias}} = \sum_{n,l} \frac{\partial L}{\partial \text{out}}$$
  
  $$\frac{\partial L}{\partial x} = \text{col2im}\left(W_{col}^T \times \frac{\partial L}{\partial \text{out}}\right)$$

##### `conv2d.py`

- **Purpose**: Implements a 2D convolutional layer that applies convolutions over inputs with multiple channels (images)
- **Main class**: `Conv2d`
- **Main features**:
  - Support for 4D inputs `(batch_size, channels, height, width)`
  - 2D convolutional kernel with configurable size `(KH, KW)`
  - Configurable stride and padding in both dimensions
  - Multiple padding modes: `zeros`, `reflect`, `replicate`, `circular`
  - Weight initialization using `DEFAULT_UNIFORM_INIT_MAP` from `config.py`
  - Efficient implementation using `im2col` transformation and matrix multiplication
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - Uses `Parameters` for trainable weights and biases
  - Uses custom types from `novann._typing` (`KernelSize`, `Stride`, `Padding`, etc.)
  - Initialized with `DEFAULT_UNIFORM_INIT_MAP["relu"]` by default (configurable)
  - Compatible with `BatchNorm2d` for batch normalization
- **Usage in the framework**:
  - For image processing in convolutional networks
  - As a main component in CNN architectures for computer vision
  - Used in combination with pooling and normalization layers
- **Technical details**:

  **im2col transformation** (convolution as matrix multiplication):
  
  $$\text{col} = \text{im2col}(x) \quad \text{(shape: } C_{in} \times K_H \times K_W \text{, } N \times H_{out} \times W_{out})$$
  
  $$W_{col} = \text{reshape}(W) \quad \text{(shape: } C_{out} \text{, } C_{in} \times K_H \times K_W)$$
  
  $$\text{out} = W_{col} \times \text{col} \quad \text{(shape: } C_{out} \text{, } N \times H_{out} \times W_{out})$$

  **Output dimensions calculation**:
  
  $$H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}_H - K_H}{\text{stride}_H}\right\rfloor + 1$$
  
  $$W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}_W - K_W}{\text{stride}_W}\right\rfloor + 1$$

  **Gradients in backward pass**:
  
  $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \text{out}} \times \text{col}^T$$
  
  $$\frac{\partial L}{\partial \text{bias}} = \sum_{n,h,w} \frac{\partial L}{\partial \text{out}}$$
  
  $$\frac{\partial L}{\partial x} = \text{col2im}\left(W_{col}^T \times \frac{\partial L}{\partial \text{out}}\right)$$

  **Efficiency**: Both implementations (`Conv1d` and `Conv2d`) use the `im2col` transformation to convert the convolution operation into matrix multiplication, allowing for more efficient computation by leveraging optimized linear algebra libraries.

#### `📂 layers/📂 flatten/`

**Layer for flattening tensors, used for transition between convolutional/pooling layers and fully connected layers**

Contains:
- `flatten.py`: Implementation of the `Flatten` layer

##### `flatten.py`

- **Purpose**: Implements a layer that flattens tensors while maintaining the batch dimension, used to connect convolutional/pooling layers to fully connected layers
- **Main class**: `Flatten`
- **Main features**:
  - Flattens all dimensions except the batch dimension (axis 0)
  - No trainable parameters (only shape transformation)
  - Saves original shape for backward pass (unflattening)
  - Pure reshape operation without expensive computations
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - Does not use `Parameters` as it has no trainable parameters
  - Designed to be used between convolutional layers (`Conv2d`, `MaxPool2d`) and linear layers (`Linear`)
- **Usage in the framework**:
  - In CNN architectures to connect the output of convolutional/pooling layers to fully connected layers
  - Necessary when transitioning from multidimensional tensors (images) to vectors for classification
  - Typical example in CNNs: `Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear`
- **Technical details**:

  **Forward operation**:
  
  $$\text{flatten}(x) = \text{reshape}(x, (N, -1))$$
  
  where $N$ is the batch size and $-1$ indicates the product of all remaining dimensions.

  **Backward operation**:
  
  $$\frac{\partial L}{\partial x} = \text{reshape}\left(\frac{\partial L}{\partial \text{out}}, \text{original\_shape}\right)$$

  The layer simply saves the original shape during forward and restores it during backward, maintaining the gradient flow.

#### `📂 layers/📂 linear/`

**Implementation of fully connected layers for linear transformations**

Contains:
- `linear.py`: Implementation of the `Linear` layer for linear transformations

##### `linear.py`

- **Purpose**: Implements a linear (fully connected) layer that performs the transformation $y = xW^T + b$
- **Main class**: `Linear`
- **Main features**:
  - Complete linear transformation between feature spaces
  - Optional bias term support
  - Weight initialization using `DEFAULT_NORMAL_INIT_MAP` from `config.py`
  - Input cache for efficient gradient calculation in backward pass
  - Vectorized implementation using matrix multiplication
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - Uses `Parameters` for trainable weights (`weight`) and bias (`bias`)
  - Uses `ListOfParameters` and `InitFn` types from `novann._typing`
  - Initialized with `DEFAULT_NORMAL_INIT_MAP["default"]` by default (configurable)
  - Extensively used by `Sequential` in neural network architectures
- **Usage in the framework**:
  - As final layer in classification/regression networks
  - In fully connected networks (MLP) for transformations between hidden layers
  - After `Flatten` layers in CNN architectures
  - In combination with activation functions (`ReLU`, `Sigmoid`, `Tanh`)
- **Technical details**:

  **Forward pass**:
  
  $$y = x \cdot W^T + b$$
  
  where:
  - $x \in \mathbb{R}^{N \times D_{in}}$ (input)
  - $W \in \mathbb{R}^{D_{out} \times D_{in}}$ (weights)
  - $b \in \mathbb{R}^{1 \times D_{out}}$ (bias, optional)
  - $y \in \mathbb{R}^{N \times D_{out}}$ (output)

  **Backward pass** (gradient calculation):

  Gradient with respect to weights:
  
  $$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial y}\right)^T \cdot x$$

  Gradient with respect to bias (if exists):
  
  $$\frac{\partial L}{\partial b} = \sum_{i=1}^{N} \frac{\partial L}{\partial y_i}$$

  Gradient with respect to input:
  
  $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot W$$

  **Initialization**: By default uses `DEFAULT_NORMAL_INIT_MAP["default"]` which corresponds to `random_init_` from `core/init.py`, but can be overridden by specific initializers based on adjacent activations when used in `Sequential`.

  **Efficiency**: Uses optimized matrix multiplication (`@` operator) and caches the input to avoid recomputation during backward pass.

#### `📂 layers/📂 pooling/`

**Implementations of pooling layers (dimensional reduction) for feature extraction**

Contains two subdirectories:
- `gap/`: Global Average Pooling (1D and 2D)
- `maxpool/`: Max Pooling (1D and 2D)

##### `📂 layers/📂 pooling/📂 gap/`

**Global Average Pooling implementations for reduction to global features**

Contains:
- `global_avg_pool1d.py`: Global Average Pooling 1D for sequences
- `global_avg_pool2d.py`: Global Average Pooling 2D for images

###### `global_avg_pool1d.py`

- **Purpose**: Implements Global Average Pooling 1D that averages along the temporal dimension for each channel
- **Main class**: `GlobalAvgPool1d`
- **Main features**:
  - Reduces 3D tensors `(batch, channels, length)` to 3D `(batch, channels, 1)`
  - No trainable parameters (fixed reduction operation)
  - Saves original shape for backward pass
  - Distributes gradients uniformly in backward
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - No trainable parameters, so not included in `parameters()`
  - Typically used at the end of 1D convolutional networks
- **Usage in the framework**:
  - In 1D convolutional network architectures to reduce sequences to global features
  - As final layer before fully connected layers in sequence classification tasks
- **Technical details**:

  **Forward pass**:
  
  $$\text{output}_{n,c} = \frac{1}{L} \sum_{l=1}^{L} x_{n,c,l}$$

  **Backward pass**:
  
  $$\frac{\partial L}{\partial x_{n,c,l}} = \frac{1}{L} \cdot \frac{\partial L}{\partial \text{output}_{n,c}}$$

  where $n$ is the batch index, $c$ is the channel, $l$ is the position in the sequence, and $L$ is the original length.

###### `global_avg_pool2d.py`

- **Purpose**: Implements Global Average Pooling 2D that averages along the spatial dimensions for each channel
- **Main class**: `GlobalAvgPool2d`
- **Main features**:
  - Reduces 4D tensors `(batch, channels, height, width)` to 4D `(batch, channels, 1, 1)`
  - No trainable parameters
  - Saves original shape for backward pass
  - Distributes gradients uniformly in both spatial dimensions
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - No trainable parameters
  - Commonly used in modern CNN architectures
- **Usage in the framework**:
  - At the end of 2D convolutional networks to produce a feature vector per channel
  - To reduce dimensionality before classification in computer vision tasks
- **Technical details**:

  **Forward pass**:
  
  $$\text{output}_{n,c} = \frac{1}{H \times W} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n,c,h,w}$$

  **Backward pass**:
  
  $$\frac{\partial L}{\partial x_{n,c,h,w}} = \frac{1}{H \times W} \cdot \frac{\partial L}{\partial \text{output}_{n,c}}$$

  where $H$ and $W$ are the original spatial dimensions.

##### `📂 layers/📂 pooling/📂 maxpool/`

**Max Pooling implementations for spatial reduction while preserving dominant features**

Contains:
- `maxpool1d.py`: Max Pooling 1D for sequences
- `maxpool2d.py`: Max Pooling 2D for images

###### `maxpool1d.py`

- **Purpose**: Implements Max Pooling 1D that reduces the temporal dimension by taking the maximum value in sliding windows
- **Main class**: `MaxPool1d`
- **Main features**:
  - Reduces 3D tensors `(batch, channels, length)` to 3D `(batch, channels, length_out)`
  - Sliding window with configurable kernel size, stride, and padding
  - Support for multiple padding modes: `zeros`, `reflect`, `replicate`, `circular`
  - Efficient implementation using NumPy's `as_strided`
  - In backward: only positions with maximum value receive gradient
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - No trainable parameters
  - Uses types from `novann._typing` for consistency
- **Usage in the framework**:
  - In 1D convolutional networks for dimensionality reduction and robust feature extraction
  - After 1D convolution layers to reduce sequence length
- **Technical details**:

  **Forward pass** (for each window):
  
  $$\text{output}_{n,c,l} = \max_{k=1}^{K} x_{n,c, s \cdot l + k - p}$$
  
  where $s$ is the stride, $K$ is the kernel size, and $p$ is the padding.

  **Backward pass**:
  
  $$\frac{\partial L}{\partial x_{n,c,pos}} = \sum_{\substack{l \\ \text{pos is argmax in window } l}} \frac{1}{|\text{argmax}|} \cdot \frac{\partial L}{\partial \text{output}_{n,c,l}}$$

  The gradient is propagated only to positions that were the maximum in their window, dividing if there are multiple maxima.

  **Output length calculation**:
  
  $$L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - K}{\text{stride}}\right\rfloor + 1$$

###### `maxpool2d.py`

- **Purpose**: Implements Max Pooling 2D that reduces spatial dimensions by taking the maximum value in 2D sliding windows
- **Main class**: `MaxPool2d`
- **Main features**:
  - Reduces 4D tensors `(batch, channels, height, width)` to 4D `(batch, channels, height_out, width_out)`
  - 2D window with configurable kernel size, stride, and padding in both dimensions
  - Support for multiple padding modes
  - Implementation with `as_strided` to efficiently create 2D windows
  - Backward similar to MaxPool1d but in 2D
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - Uses `IntOrPair` types from `novann._typing` for parameters
  - No trainable parameters
- **Usage in the framework**:
  - In 2D convolutional networks for spatial reduction and invariant feature extraction
  - Typically after convolution and activation layers in CNN architectures
- **Technical details**:

  **Forward pass** (for each 2D window):
  
  $$\text{output}_{n,c,i,j} = \max_{h=1}^{K_H} \max_{w=1}^{K_W} x_{n,c, s_h \cdot i + h - p_h, s_w \cdot j + w - p_w}$$

  **Backward pass**:
  
  Similar to 1D but in 2D, propagating gradients only to maximum positions.

  **Output dimensions calculation**:
  
  $$H_{out} = \left\lfloor\frac{H_{in} + 2 \times p_h - K_H}{s_h}\right\rfloor + 1$$
  
  $$W_{out} = \left\lfloor\frac{W_{in} + 2 \times p_w - K_W}{s_w}\right\rfloor + 1$$

  **Efficiency**: Both implementations use `as_strided` to create windows without copying data. The backward pass requires loops to accumulate gradients, which could be optimized in future versions.

#### `📂 layers/📂 regularization/`

**Implementations of regularization techniques to prevent overfitting in neural networks**

Contains:
- `dropout.py`: Implementation of the `Dropout` layer for regularization by randomly turning off neurons

##### `dropout.py`

- **Purpose**: Implements the Dropout regularization technique that randomly turns off neurons during training to prevent overfitting
- **Main class**: `Dropout`
- **Main features**:
  - During training: randomly turns off input elements with probability `p` and scales the remaining ones to preserve expected activations
  - During evaluation: acts as identity (no dropout)
  - Maintains consistency of expected activations through scaling `1/(1-p)`
  - Clears internal mask after backward to avoid references between batches
  - Supports dropout probability in range `[0.0, 1.0)`
- **Integration**:
  - Inherits from `Layer` from `novann.module`
  - No trainable parameters
  - Overrides `train()` and `eval()` methods to correctly manage the mode
  - Compatible with all framework layers that follow the standard interface
- **Usage in the framework**:
  - Inserted between layers in deep networks for regularization
  - Typically used after `Linear` or `Conv` layers and before activations
  - Useful for preventing overfitting in networks with high capacity
- **Technical details**:

  **Forward pass (Training Mode)**:

  For each element $x_i$ of the input:
  
  1. Generate binary mask:
     $$m_i \sim \text{Bernoulli}(1-p)$$

  2. Apply dropout and scaling:
     $$y_i = \frac{x_i \cdot m_i}{1-p}$$

  **Forward pass (Evaluation Mode)**:
  
  $$y_i = x_i \quad \text{(no changes)}$$

  **Backward pass (Training Mode)**:
  
  $$\frac{\partial L}{\partial x_i} = \frac{m_i}{1-p} \cdot \frac{\partial L}{\partial y_i}$$

  **Statistical properties**:
  
  During training:
  $$E[y_i] = E\left[\frac{x_i \cdot m_i}{1-p}\right] = x_i \cdot \frac{E[m_i]}{1-p} = x_i \cdot \frac{1-p}{1-p} = x_i$$

  This ensures that the expected activation remains the same during training and evaluation.

  **Practical implementation**:
  - The mask is generated using `np.random.rand()` and converted to boolean
  - It is stored with the same dtype as the input for efficiency
  - It is cleared after backward to free memory and avoid cross-references
  - Dimensions are maintained in all operations

  **Performance considerations**:
  - The scaling `1/(1-p)` is applied during forward and backward for consistency
  - In evaluation mode, there is no computational overhead
  - Random generation introduces some overhead but is essential for the regularization effect


This hierarchy enables a modular design where each component follows a consistent interface, facilitating composition and training of complex neural networks.

### `📂 optim/`

**Optimizer Implementations for Neural Network Training**

Contains:
- `adam.py`: Adam (Adaptive Moment Estimation) optimizer with coupled weight decay
- `adamw.py`: AdamW optimizer with decoupled weight decay
- `rmsprop.py`: RMSprop (Root Mean Square Propagation) optimizer with decoupled weight decay
- `sgd.py`: SGD (Stochastic Gradient Descent) optimizer with momentum and gradient clipping

#### `adam.py`

- **Purpose**: Implements the Adam (Adaptive Moment Estimation) optimizer, combining the advantages of AdaGrad and RMSProp with first- and second-order moments
- **Main Class**: `Adam`
- **Main Features**:

- Adaptive estimations of first- and second-order moments
- Bias correction for moments in the first iterations
- Support for L2 weight decay **coupled to the gradient**
- Configurable beta coefficients for moment decay rates
- Automatic exclusion of BatchNorm parameters from weight decay
- epsilon term for numerical stability in division

- **Integration**:
- Operates on lists of `Parameters` from `novann.module`
- Uses the `ListOfParameters` type from `novann._typing`
- Automatically excludes `gamma` and `beta` parameters from BatchNorm from weight decay
- Compatible with all models that implement the `parameters()` method

- **Use in the framework**:
- Classic optimizer for many deep learning problems
- Suitable for networks with complex architectures and a large number of parameters
- Used in classification and regression examples in the framework
- **Technical details**:

**Algorithm of update**: 

For each parameter $\theta$ in step $t$: 

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$ 

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$ 

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$ 

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$ 

$$\theta _{t+1} = \theta_t - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$ 

where: 
- $\eta$: Learning rate (`lr`) 
- $\beta_1, \beta_2$: Decay coefficients (`betas`)

- $g_t$: Gradient at step $t$

- $\epsilon$: Numerical stability term (`eps`)

**Coupled weight decay** (excluding BatchNorm parameters):

$$g_t \leftarrow g_t + \lambda \theta_t$$

The weight decay is applied **directly to the gradient** before the moment update, coupling regularization with adaptive optimization.

#### `adamw.py`

- **Purpose**: Implements the AdamW optimizer, which improves upon Adam using **decoupled** weight decay, separating regularization from adaptive updating.
- **Main Class**: `AdamW`
- **Main Features**:
  - Adaptive moment estimates identical to Adam
- **Decoupled** weight decay applied directly to the parameters (not the gradient)
  - Bias correction for moments in the first iterations
  - Automatic exclusion of BatchNorm parameters from weight decay
  - Better generalization than Adam in many practical cases
  - Configurable `beta` coefficients for moment decay rates
- **Integration**:
  - Operates on `Parameters` lists from `novann.module`
  - Uses `ListOfParameters` and `BetaCoefficients` types from `novann._typing`
  - Automatic recognition of BatchNorm parameters by name (`gamma`, `beta`)
  - Compatible with the framework's standard optimizer interface
- **Usage within the framework**:

- **Recommended over Adam** for most modern use cases

  - Provides better regularization without affecting adaptive dynamics
- **Technical Details**:
**Update Algorithm**:

  For each parameter $\theta$ in step $t$:

**Update Moments** (identical to Adam):

  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

  $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Adaptive Update**:

  $$\theta_{t+1} = \theta_t - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

  **Decoupled Weight Decay** (applied **after** the adaptive update):

  $$\theta_{t+1} \leftarrow \theta_{t+1} - \eta \cdot \lambda \cdot \theta_t$$

  where $\lambda$ is the weight decay coefficient.


**Key difference from Adam**:

- **Adam**: Weight decay is coupled to the gradient → affects adaptive dynamics

- **AdamW**: Weight decay is applied directly to the parameters → pure regularization without interfering with adaptation

This separation allows weight decay to function as **true regularization** independent of the gradient magnitude, improving generalization.

#### `rmsprop.py`

- **Purpose**: Implements the RMSprop optimizer, which maintains a moving average of squared gradients to adapt the learning rate per parameter.
- **Main Class**: `RMSprop`
- **Main Features**:
  - Moving average of squared gradients to adapt the step size per parameter.
  - Support for decoupled L2 weight decay.
  - Automatic exclusion of BatchNorm parameters from weight decay.
  - Configurable decay coefficient for the moving average.
  - Simple and efficient implementation.
- **Integration**:
  - Operates on `Parameters` lists from `novann.module`.
  - Uses the `ListOfParameters` type from `novann._typing`.
  - Automatic recognition of BatchNorm parameters by name (`gamma`, `beta`).
  - Compatible with the framework's standard optimizer interface.
- **Usage within the framework**:
  - Alternative to Adam/AdamW For problems where more conservative adaptations are preferred
  - Option available in the training examples
- **Technical Details**:

**Update Algorithm**:

  For each parameter $\theta$:

  $$E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2$$

  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

  where:

- $\eta$: Learning rate (`lr`)

- $\beta$: Decay coefficient for the moving average

- $g_t$: Gradient at step $t$

- $\epsilon$: Numerical stability term

**Decoupled weight decay** (applied after the update): 

$$\theta _{t+1} \leftarrow \theta _{t+1} - \eta \cdot \lambda \cdot \theta_t$$

#### `sgd.py`

- **Purpose**: Implements the SGD (Stochastic Gradient Descent) optimizer with momentum, weight decay, and global gradient clipping
- **Main Class**: `SGD`
- **Main Features**:
  - Classical stochastic gradient descent with optional momentum (Polyak momentum)
  - Global gradient clipping to prevent gradient explosions
  - Support for L2 weight decay **coupled to the gradient**
  - Automatic exclusion of BatchNorm parameters from weight decay
  - Efficient implementation with velocity buffers for momentum
- **Integration**:
  - Operates on `Parameters` lists from `novann.module`
  - Uses the `ListOfParameters` type from `novann._typing`
  - Gradient clipping system that considers the overall norm of all gradients
  - Compatible with the framework's training interface
- **Use within the framework**:
  - Standard optimizer for problems where simplicity and control are preferred Fine-tuning
  - Useful for fine-tuning and small dataset problems
  - Gradient clipping is especially useful for recurrent networks
- **Technical Details**:

**Update Algorithm** (with momentum):

  For each parameter $\theta$:

  $$v_t = \beta v_{t-1} - \eta g_t$$

  $$\theta_{t+1} = \theta_t + v_t$$

  Without momentum:

  $$\theta_{t+1} = \theta_t - \eta g_t$$

**Global Gradient Clipping**:

  $$\text{total\_norm} = \sqrt{\sum_i \|g_i\|^2}$$

  $$\text{clip\_coef} = \min\left(1.0, \frac{\text{max\_grad\_norm}}{\text{total\_norm} + 1e-6}\right)$$

  $$g_i \leftarrow g_i \cdot \text{clip_coef}$$

**Coupled weight decay** (applied to the gradient):

$$g_t \leftarrow g_t + \lambda \theta_t$$

**Common features of optimizers**:
  - They all implement `step()` to update parameters and `zero_grad()` to clean gradients
  - They exclude `gamma` and `beta` parameters from BatchNorm from the weight decay (detected by name)
  - They properly handle parameters without a gradient (`grad is None`)
  - They are iterable over materialized parameter lists
- **Adam and SGD**: Use coupled weight decay (applied to the gradient)
- **AdamW and RMSprop**: Use decoupled weight decay (applied directly to the parameters)

### `📂 utils/`

**Utilities for data handling, dataset loading, logging, visualizations, training and gradient checking**

Contains:
- `data/`: Utilities for data handling and preprocessing
- `datasets/`: Functions for loading common computer vision datasets
- `decorators/`: Decorators for timing and profiling
- `gradient_checking/`: Utilities for numerical gradient verification
- `log_config/`: Logging system configuration
- `train/`: Model training function
- `visualizations/`: Utilities for results and metrics visualization

#### `📂 utils/📂 data/`

**Utilities for data handling and preprocessing**

Contains:
- `dataloader.py`: `DataLoader` class for iterating over datasets in minibatches
- `preprocessing.py`: Functions for normalization and feature/label separation

##### `dataloader.py`

- **Purpose**: Implements an iterable DataLoader that allows traversing a dataset in minibatches, with support for shuffling
- **Main class**: `DataLoader`
- **Main features**:
  - Supports configurable fixed batch size
  - Can shuffle data at the start of each epoch
  - Implements Python iterator protocol with internal iterator `_Iter`
  - Automatically calculates number of batches per epoch via `__len__`
  - Properly handles the last batch which may be smaller
- **Integration**:
  - Used in example scripts to provide data to models during training and evaluation
  - Compatible with metric functions (`accuracy`, `binary_accuracy`, `r2_score`) that require a batch iterator
  - `Loader` type defined in `novann/_typing.py` refers to this class
- **Usage in the framework**:
  - Used in training loops to obtain data batches
  - Also used in evaluation to compute metrics over the entire dataset
  - Allows efficient iteration over large datasets without fully loading them into memory

##### `preprocessing.py`

- **Purpose**: Preprocessing functions for normalization and feature/label separation
- **Main functions**:
  - `normalize`: Normalizes data by subtracting mean and dividing by standard deviation
  - `split_features_and_labels`: Separates a pandas DataFrame into feature and label arrays
- **Implementation**:

  **Normalization**:
  
  $$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$
  
  **Feature/label separation**:
  - Automatically detects if "label" column exists in DataFrame
  - If not, assumes the first column contains labels
- **Integration**:
  - Used by dataset loading functions (`load_fashion_mnist_data` and `load_mnist_data`)
  - Normalization uses training set statistics to avoid data leakage
  - Compatible with pandas DataFrames loaded from CSV
- **Usage in the framework**:
  - Data preprocessing before training
  - Feature normalization to stabilize training
  - Separation of data into features (X) and labels (y)

#### `📂 utils/📂 datasets/`

**Functions for loading common computer vision datasets**

Contains:
- `fashion.py`: Loading of Fashion-MNIST dataset
- `mnist.py`: Loading of MNIST dataset

##### `fashion.py`

- **Purpose**: Loads Fashion-MNIST dataset from CSV files and optionally normalizes and transforms it to 4D tensors
- **Main function**: `load_fashion_mnist_data`
- **Main features**:
  - Loads training, test and validation data from specified paths
  - Supports normalization using mean and standard deviation of training set
  - Can convert data to 4D format `(N, 1, 28, 28)` for 2D convolutional layers
  - Uses pandas with pyarrow backend for memory efficiency
  - Error handling with appropriate logging
  - Returns tuples `(x_train, y_train), (x_test, y_test), (x_val, y_val)`
- **Integration**:
  - Uses `split_features_and_labels` and `normalize` from `utils/data/preprocessing.py`
  - Uses constants from `core/constants.py` for default paths (`EXPORTATION_FASHION_TRAIN_DATA_PATH`, etc.)
  - Returns tuples of type `TrainTestEvalSets` defined in `novann/_typing.py`
  - Logs events with logger from `novann/utils/log_config/`
- **Usage in the framework**:
  - Provides data for image classification examples
  - Used in comparisons with PyTorch and benchmark experiments

##### `mnist.py`

- **Purpose**: Loads MNIST dataset from CSV files and optionally normalizes and transforms it to 4D tensors
- **Main function**: `load_mnist_data`
- **Main features**:
  - Similar functionality to `load_fashion_mnist_data` but for MNIST dataset
  - Optional normalization with training set statistics
  - Transformation to 4D `(N, 1, 28, 28)` for 2D convolutions
  - Uses pandas with pyarrow for efficient loading
  - Robust error handling with logging
- **Integration**:
  - Uses same preprocessing functions as `fashion.py`
  - Uses constants from `core/constants.py` for MNIST paths (`EXPORTATION_MNIST_TRAIN_DATA_PATH`, etc.)
  - Same return type `TrainTestEvalSets` and error handling
- **Usage in the framework**:
  - Provides MNIST dataset for classification examples
  - Classic dataset for framework testing and demonstrations

#### `📂 utils/📂 decorators/`

**Decorators for cross-cutting functionality like timing and profiling**

Contains:
- `timing.py`: `@chronometer` decorator for measuring function execution time

##### `timing.py`

- **Purpose**: Provides the `@chronometer` decorator to automatically and non-intrusively measure and log function execution time
- **Main decorator**: `@chronometer`
- **Main features**:
  - Measures execution time with high precision using `time.perf_counter()`
  - Intelligent time formatting: adapts units from nanoseconds to hours
  - Contextual emoji usage (⚡ for fast, ⏱️ for normal, 🐢 for slow)
  - Preserves original function metadata with `@wraps(func)`
  - Does not modify the result of the decorated function
  - Automatic logging using framework's logging system
- **Integration**:
  - Imports and uses `logger` from `novann/utils/log_config/` to log times
  - Generic decorator that can be applied to any callable function
  - Used by `train()` function from `utils/train/train.py` to measure training time
- **Usage in the framework**:
  - Profiling critical functions for performance optimization
  - Measuring training time in examples and experiments
  - Performance debugging during development
- **Technical details**:

  **Time formatting algorithm**:
  
  - < 1 microsecond: displays in nanoseconds (ns)
  - < 1 millisecond: displays in microseconds (μs) 
  - < 1 second: displays in milliseconds (ms)
  - < 1 minute: displays in seconds with 2 decimals (s)
  - < 1 hour: displays in minutes and seconds (m s)
  - ≥ 1 hour: displays in hours, minutes and seconds (h m s)

  **Implementation**:
  ```python
  # Usage example
  @chronometer
  def slow_function():
      # code that takes time
      pass
  ```

#### `📂 utils/📂 gradient_checking/`

**Utilities for numerical gradient verification using finite differences**

Contains:
- `numerical.py`: Functions to compute numerical gradients using central differences

##### `numerical.py`

- **Purpose**: Implements functions to compute numerical gradients using finite differences, used to verify correctness of backpropagation implementations
- **Main functions**:
  - `numeric_grad_elementwise`: Elementwise numerical gradient for vector functions
  - `numeric_grad_scalar_from_softmax`: Specific numerical gradient for softmax + scalar loss
  - `numeric_grad_scalar_wrt_x`: Generic numerical gradient for scalar losses
  - `numeric_grad_wrt_param`: Numerical gradient with respect to layer parameters
- **Common features**:
  - All use central differences for greater precision: $\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$
  - Restore original values after each perturbation
  - Iterate over individual elements using `np.nditer` to handle multidimensional arrays
  - Configurable `eps` term for finite difference step size
- **Integration**:
  - Used in unit tests to verify gradients of activations, linear and convolutional layers
  - Independent of specific implementations, work with any function following the appropriate interface
  - Not used during normal training, only for debugging and testing
- **Technical details**:

  **Central differences** (base formula):
  
  $$\frac{\partial f}{\partial x_i} \approx \frac{f(x_i + \epsilon) - f(x_i - \epsilon)}{2\epsilon}$$

  **`numeric_grad_elementwise`**:
  - For vector functions $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$
  - Calculates $\frac{\partial f_j}{\partial x_i}$ for each $i,j$
  - Used to verify gradients of activation functions

  **`numeric_grad_scalar_from_softmax`**:
  - For $L = \sum(\text{softmax}(x) \cdot G)$ where $G$ is a weight matrix
  - Calculates $\frac{\partial L}{\partial x_i}$
  - Specific for testing softmax + cross-entropy

  **`numeric_grad_scalar_wrt_x`**:
  - For $S = \sum(\text{forward_fn}(x) \cdot G)$
  - Calculates $\frac{\partial S}{\partial x_i}$
  - Generic version for any forward function

  **`numeric_grad_wrt_param`**:
  - For $S = \sum(\text{layer.forward}(x) \cdot G)$
  - Calculates $\frac{\partial S}{\partial p_i}$ where $p_i$ are layer parameters
  - Perturbs parameter data (`p.data`) and restores afterwards

#### `📂 utils/📂 log_config/`

**Logging system configuration for the framework**

Contains:
- `logger.py`: Implementation of the custom `Logger` with singleton pattern

##### `logger.py`

- **Purpose**: Provides a unified logging system for the entire framework with support for console and file
- **Main class**: `Logger` (implements singleton pattern)
- **Main features**:
  - Singleton pattern: only one instance in the entire application
  - Support for multiple levels: DEBUG, INFO, WARNING, ERROR
  - Handlers for console and file (configurable)
  - Customizable format with timestamp, level, logger name and message
  - Methods for logging with additional data via `**kwargs`
  - Dynamic logging level change
- **Integration**:
  - Imported and used by all framework modules that need logging
  - Uses constants from `core/constants.py` for default configuration (`LOG_FILE`, `LOGGER_DEFAULT_FORMAT`, `LOGGER_DATE_FORMAT`)
  - Global `logger` instance created at module level for easy access
- **Usage in the framework**:
  - Debugging during development and testing
  - Event logging during training (loss, metrics, errors)
  - Tracking parameter initialization in `Sequential`
  - Error handling in dataset loading

#### `📂 utils/📂 train/`

**Model training function**

Contains:
- `train.py`: `train()` function that implements the complete training cycle

##### `train.py`

- **Purpose**: Provides a high-level function to train models in a simple and configurable way
- **Main function**: `train()` (decorated with `@chronometer` for time measurement)
- **Main features**:
  - Complete training cycle with epochs and batches
  - Support for periodic validation metrics
  - Configurable progress logging
  - Automatic handling of model train/eval modes
  - Integration with any optimizer and loss function of the framework
  - Decorated with `@chronometer` to measure execution time
- **Integration**:
  - Uses `DataLoader` from `utils/data/` to iterate over data
  - Expects a `Sequential` model from `model/nn.py`
  - Compatible with any `Optimizer` (Adam, SGD, RMSprop) and `LossFunc` (CrossEntropyLoss, MSE, etc.)
  - Uses the `logger` from `utils/log_config/` for progress logging
  - `@chronometer` decorator from `utils/decorators/` for timing
- **Usage in the framework**:
  - Main function to train models in examples and experiments
  - Simplifies training code by eliminating the need to write manual loops
  - Provides a standard entry point for training

## Common Usage Patterns

Let's say we want to make an image classifier for the _fashion-mnist_ dataset, the normal workflow would be:

```python
# 1. import necessary tools
from novann.model import Sequential
from novann.optim import Adam
from novann.utils.data import DataLoader
from novann.losses import CrossEntropyLoss
from novann.metrics import accuracy
from novann.utils.datasets import load_fashion_mnist_data
from novann.layers import (
    Conv2d,
    Linear, 
    ReLU,
    Flatten
    BatchNorm2d, 
    MaxPool2d
)

# 2. load data to use
(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_fashion_mnist_data(
    tensor4d=True, do_normalize=True
)

# 3. define the model
model = Sequential(
    Conv2d(1, 32, 3, padding=1, bias=False),
    BatchNorm2d(32),
    ReLU(),
    MaxPool2d(2, 2),
    Conv2d(32, 64, 3, padding=1, bias=False),
    BatchNorm2d(64),
    ReLU(),
    MaxPool2d(2, 2),
    Linear(64 * 8 * 8, 10) # -> 10 classes (from 0 to 9)
)

# if you print the model you'll see something like
print(model)
"""
Sequential(
  (0): Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (1): BatchNorm2d(num_features=32, momentum=0.1, eps=1e-05)
  (2): ReLU()
  (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
  (4): Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (5): BatchNorm2d(num_features=64, momentum=0.1, eps=1e-05)
  (6): ReLU()
  (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
  (8): Linear(in_features=4096, out_features=10, bias=True)
)
"""

# 4. Set optimizer and hyperparameterss
lr = 1e-3
batch_size = 128
epochs = 10
optimizer = Adam(
    model.parameters() # Model parameters are passed
    lr=lr, 
    weight_decay=1e-5
    betas=(0.9,0.999)
)

# 5. define the loss function
loss_fn = CrossEntropyLoss()

# 6. Create data loaders
train_loader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(x_test, y_test, batch_size=batch_size, shuffle=False)

# 7. make a training loop (or call the train function)

model.train() # set to training mode

for epoch in range(epochs):
    for input, label in train_loader:
        # set gradients to zero
        optimizer.zero_grad()

        # calculate forward pass
        logits = model(input)

        # calculate total loss of mini batch and gradient
        loss, grad = loss_fn(logits, label)

        # execute backward pass
        model.backward(grad)

        # update parameters once gradients are calculated
        optimizer.step()
    
    model.eval() # set to evaluation mode
    acc = accuracy(model, val_loader)

    model.train() # Set back to training mode

    # (Optional) print results
    print(f"Epoch {epoch + 1}/{epochs}, loss: {loss:.4f}, validation accuracy: {acc:.3f}")

# 8. final evaluation with test set
model.eval()
acc = accuracy(model, test_loader)
print(f"Test accuracy {acc:.3}")
```

## 🛠️ Technologies Used

The **NovaNN** framework is built using the following main technologies and libraries:

- **Language**: Python >= 3.14
- **Dependency Management**: Poetry (for package management and virtual environments)
- **Main Libraries**:
    - `numpy`: Efficient numerical operations and multidimensional arrays
    - `pandas`: Handling and analyzing tabular data (for loading datasets)
    - `matplotlib`: Visualizing graphs and results
    - `seaborn`: Aesthetic enhancement of statistical visualizations
    - `scikit-learn`: Classical Machine Learning tools and utilities
    - `pyarrow`: Efficient backend for pandas DataFrames (reduces memory usage)
- **Development Tools**:
    - `pytest`: Unit testing framework
    - `pytest-cov`: Code coverage in tests
    - `python-dotenv`: Environment variable management from `.env` files
    - `ipykernel`: Jupyter kernel for notebooks
    - `black`: Code formatter for consistent style

## 📦 Installation

NovaNN uses **Poetry** for dependency management and packaging. Follow these steps to set up your environment:

### 1. Clone the repository

```bash
git clone https://github.com/JOSE-MDG/NovaNN.git
cd NovaNN
```

### 2. Install Poetry (if you don't already have it installed)
- Windows (PowerShell):

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

- Linux/macOS:

```bash
# With curl
curl -sSL https://install.python-poetry.org | Python 3 -

# With pipx
pipx install poetry
```

- Add Poetry to the PATH (if necessary):

```bash
# On Linux/macOS, add to ~/.bashrc or ~/.zshrc:
export PATH="$HOME/.local/bin:$PATH"
```

### 3. Install project dependencies

```bash
# Install all dependencies (including development dependencies)
poetry install
```

### 4. Activate the virtual environment

```bash
# Install the shell plugin
`poetry self add poetry-plugin-shell

# # Activate the shell with the virtual environment
`poetry shell

# Alternatively, run commands directly without activating the shell:
`poetry run python examples/binary_classification.py
```

### 5. Configure environment variables

Create a file Create a .env file in the project root directory with the following variables (adjust the paths according to your configuration):

```env
# Paths for Fashion-MNIST
FASHION_TRAIN_DATA_PATH=<YOUR PATH>/NovaNN/data/FashionMnist/fashion-mnist_train.csv
EXPORTATION_FASHION_TRAIN_DATA_PATH=<YOUR PATH>/data/FashionMnist/fashion_train_ready.csv
FASHION_VALIDATION_DATA_PATH=<YOUR PATH>/data/FashionMnist/fashion_validation_ready.csv
FASHION_TEST_DATA_PATH=<YOUR PATH>/data/FashionMnist/fashion-mnist_test.csv

# Paths for MNIST
MNIST_TRAIN_DATA_PATH=<YOUR PATH>/data/FashionMnist/fashion-mnist_test.csv PATH>/data/Mnist/mnist_train.csv
EXPORTATION_MNIST_TRAIN_DATA_PATH=<YOUR PATH>/data/Mnist/mnist_train_ready.csv
MNIST_VALIDATION_DATA_PATH=<YOUR PATH>/data/Mnist/mnist_validation_ready.csv
MNIST_TEST_DATA_PATH=<YOUR PATH>/data/Mnist/mnist_test.csv

# Logging configuration
LOG_FILE=<YOUR PATH>/logs/nova_nn.log
LOGGER_DEFAULT_FORMAT=%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s # This can be any format you want
LOGGER_DATE_FORMAT=%Y-%m-%d %H:%M:%S
```

### 6. Run examples

```bash
# Binary classification
poetry run python examples/binary_classification.py

# Multiclass classification
poetry run python examples/multiclass_classification.py

# Convolutional neural networks
poetry run python examples/conv_example.py

# Regression
poetry run python examples/regression.py
```

### 7. Run all tests

```bash
# All tests
poetry run pytest tests/

# Specific tests with coverage
poetry run pytest tests/ --cov=novann --cov-report=term-missing

# Verbose tests
poetry run pytest tests/ -v
```


## 🧪 Testing
The framework includes a complete suite of unit tests in the `tests/` directory (./tests/) that verify the correct implementation of all components. For more information, see [Unit Tests](./tests/README.en.md).

## 🤝 Contribution

Contributions are welcome and appreciated. NovaNN is an open-source educational project that benefits from the community.

### **How ​​to Contribute?**

1. **Fork the repository** on GitHub
2. **Create a branch for your feature** (`git checkout -b feature/new-feature`)
3. **Commit your changes** (`git commit -m 'Add new feature X'`)
4. **Push to the branch** (`git push origin feature/new-feature`)
5. **Open a Pull Request** on GitHub with a clear description of the changes

### **Priority Contribution Areas**

- 🐛 **Bug reporting and fixing**: Test the framework in different scenarios
- 💡 **New layers and features**: Implementations of recent papers
- 📚 **Documentation improvement**: Additional examples, tutorials, code documentation
- 🧪 **Unit tests**: Increase coverage and edge cases
- ⚡ **Optimizations of Performance**: Improvements in NumPy implementations
- 🔧 **Development Tools**: Utility scripts, visualizations

### **Style and Quality Guidelines**

- **Code**: Follow existing conventions and use Black for formatting
- **Tests**: Include unit tests for new features
- **Documentation**: Update docstrings and README if necessary
- **Type Hints**: Use type hints consistently
- **Commits**: Descriptive messages in English or Spanish

### **Review Process**

- PRs will be reviewed by the lead maintainer
- Passing tests and maintained coverage are expected
-
- Changes can be requested before merging

### **Issue Reporting**

When reporting a bug or requesting a feature:

- **Clear and descriptive title**
- **Detailed description** of the problem or request
- **Steps to reproduce** (for bugs)
- **Expected vs. Actual Behavior**
- **Environment**: Python version, operating system, NovaNN version
- **Minimum Sample Code** for Reproduction
- **Relevant Logs** (use the framework's logger)

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

**MIT License Summary:**
- Free software to use, copy, modify, merge, publish, and distribute
- Can be used for commercial purposes
- The license includes original copyrights
- No warranty and the authors are not liable for damages

## 👤 Author and Maintainer

**Juan José** - Developer & Machine Learning Engineer (16 years old)

- GitHub: [https://github.com/JOSE-MDG](https://github.com/JOSE-MDG)
- Email: josepemlengineer@gmail.com

**About me**: At just 16 years old, I built **NovaNN** from scratch as an educational project to demonstrate my passion and deep understanding of deep learning. This framework represents months of self-directed study, experimentation, and dedication, mathematically implementing each algorithm from the original papers.

**Acknowledgments:**
- Inspired by PyTorch and other deep learning frameworks
- Open source community for shared tools and knowledge
- Research papers that support the implementations