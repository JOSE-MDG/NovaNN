```
███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗███╗   ██╗
████╗  ██║██╔═══██╗██║   ██║██╔══██╗████╗  ██║████╗  ██║
██╔██╗ ██║██║   ██║██║   ██║███████║██╔██╗ ██║██╔██╗ ██║
██║╚██╗██║██║   ██║██║   ██║██╔══██║██║╚██╗██║██║╚██╗██║
██║ ╚████║╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║██║ ╚████║
╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝
                NovaNN — Deep Learning Framework
```

![version](https://img.shields.io/badge/version-1.0.0-blue)
![python](https://img.shields.io/badge/python-3.14-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![tests](https://img.shields.io/badge/tests-pytest-orange)

## 🌐 Available languages

- 🇬🇧 [English](README.en.md)
- 🇪🇸 [Español](README.md)

This mini framework provides tools and examples for creating **MLP** neural networks along with modules that support and enhance network training. This project aims to demonstrate a solid understanding and mastery of how these networks work, inspired by popular deep learning frameworks like **PyTorch** and **TensorFlow**, especially **PyTorch** which was the main inspiration for this project.

**Clarification**: This mini framework aims to demonstrate solid foundations and knowledge about how neural networks work, Deep Learning, Machine Learning, mathematics, software engineering, best practices, unit tests, modular design, and data preprocessing.

## Introduction

- This project has a completely **modular** structure; it includes a directory called `examples/` with examples of **binary classification**, **multiclass classification**, and **regression** showing how to use the tools provided by this mini framework.

- The `data/` directory contains datasets like _Fashion-MNIST_ and _MNIST_ where _Fashion-MNIST_ was used to compare the project's performance with another framework and _MNIST_ was used for a normal classification usage example in the `examples/` directory.

- A review, preprocessing, and prior data split was performed in `notebooks/exploration.ipynb` where datasets were visualized and the validation set was partitioned for both.

- The `src/` module is the main module that contains all the parts and/or tools that make up this mini framework. It has a centralized structure where `core/config.py` stores and loads environment variable values so they can be accessible by the rest of the modules, avoiding the need to load them in each script.

- The performance of **NovaNN Framework** was evaluated against the popular deep learning framework **PyTorch** in a classification task with the _Fashion-MNIST_ dataset, using exactly the same dataset and hyperparameters for both tests. To obtain the comparison results, metrics like accuracy and loss were saved in `json` format.

  - **[main.py](main.py)**: This file implements the training code and the network structure to be used for the comparison.
  - **[pytorch_comparison](https://colab.research.google.com/drive/1APfspox9ONmDWL0jFXmndHZ70UPjr9Mn?usp=sharing)**: The notebook contains the PyTorch version training code, which performs the same procedure as the script.

### Comparison Results:

Once the results were obtained, a script ([visualization.py](visualization.py)) was created to graph the results in a more presentable way.

![comparison](images/comparison.png)

## Project Structure

[Structure file](FileTree_NeuralNetwork.md)

```
📁 Neural Networks
├── 📁 data
│   ├── 📁 FashionMnist
│   └── 📁 Mnist
├── 📁 examples
│   ├── 🐍 binary_classification.py
│   ├── 🐍 multiclass_classification.py
│   └── 🐍 regresion.py
├── 📁 images
│   └── 🖼️ comparison.png
├── 📁 logs
├── 📁 notebooks
│   └── 📄 exploration.ipynb
├── 📁 src
│   ├── 📁 core
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 config.py
│   │   ├── 🐍 dataloader.py
│   │   ├── 🐍 init.py
│   │   └── 🐍 logger.py
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
│   │   │   └── 🐍 batch_normalization.py
│   │   ├── 📁 linear
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 linear.py
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
│   ├── 🐍 __init__.py
│   └── 🐍 utils.py
├── 📁 tests
│   ├── 📁 activations
│   │   ├── 🐍 test_leaky_relu.py
│   │   ├── 🐍 test_relu.py
│   │   ├── 🐍 test_sigmoid.py
│   │   ├── 🐍 test_softmax.py
│   │   └── 🐍 test_tanh.py
│   ├── 📁 batch_norm
│   │   └── 🐍 test_batch_norm.py
│   ├── 📁 dataloader
│   │   └── 🐍 test_dataloader.py
│   ├── 📁 initializers
│   │   └── 🐍 test_init.py
│   ├── 🐍 test_dropout_regularization.py
│   ├── 🐍 test_linear_layer.py
│   └── 🐍 test_sequential_module.py
├── ⚙️ .gitignore
├── 📝 FileTree_NeuralNetwork.md
├── 📝 README.en.md
├── 📝 README.es.md
├── 🐍 main.py
├── 📄 requirements.txt
└── 🐍 visualization.py
```

## Structure of the `src/` module and subdirectories

Here we will explain in detail what each submodule and its files do.

### `core/`

**Centralizes the essential functions of the project**

#### `config.py`

- **Purpose**: Global configurations and environment variables
- **Environment variables**:
  - Dataset paths: `FASHION_TRAIN_DATA_PATH`, `MNIST_TRAIN_DATA_PATH`, etc.
  - Logging configuration: `LOG_FILE`, `LOGGER_DEFAULT_FORMAT`, etc.
- **Initialization dictionaries**:
  - `DEFAULT_NORMAL_INIT_MAP`: Initialization functions with normal distribution (Xavier/Kaiming)
  - `DEFAULT_UNIFORM_INIT_MAP`: Initialization functions with uniform distribution (Xavier/Kaiming)
  - **Keys**: Activation names (`relu`, `leakyrelu`, `tanh`, `sigmoid`, `default`)
- **Integration with activations**: The keys in the initialization maps correspond to the names of activations in `layers/activations/`
- **Usage in linear layers**: The initialization maps are used by `Linear.reset_parameters()`
- **Complete integration**: The initialization maps feed `Linear.reset_parameters()` which creates `Parameters` objects
- **Complete ecosystem**: The initialization maps feed the Module → Layer → Linear inheritance system
- **Usage in Sequential**: The initialization maps are used by `Sequential._apply_initializer_for_linear_layers()` for automatic initialization

#### `dataloader.py`

- **Class `DataLoader`**:
  - **Purpose**: Iterable data loader for mini-batches
  - **Features**:
    - Supports data shuffling
    - Configurable batch size (`batch_size`)
    - Implements Python iterator protocol
  - **Complete integration**: Used in all examples for efficient batch management
  - **Main methods**:
    - `__init__`: Initializes with `x` (features) and `y` (labels) arrays
    - `__iter__`: Creates iterator for one epoch
    - `__len__`: Returns number of batches per epoch

#### `init.py`

- **Purpose**: Utilities for weight initialization
- **Usage in layers**: Initialization functions are used by linear layers based on the following activation
- **Functions**:
  - `calculate_gain(nonlinearity, param)`: Calculates gain for a nonlinearity
  - `xavier_normal_(shape, gain)`: Xavier normal initialization
  - `xavier_uniform_(shape, gain)`: Xavier uniform initialization
  - `kaiming_normal_(shape, a, nonlinearity, mode)`: Kaiming normal initialization
  - `kaiming_uniform_(shape, a, nonlinearity, mode)`: Kaiming uniform initialization
  - `random_init_(shape, gain)`: Small random initialization (conservative default)

#### `logger.py`

- **Class `Logger`**:
  - **Purpose**: Custom logger with multiple levels
  - **Features**:
    - Console and/or file logging
    - Customizable formatting
    - Support for additional data via `**kwargs`
  - **Methods**:
    - `info`, `debug`, `warning`, `error`: Logging at different levels
    - `_create_console_handler`, `_create_file_handler`: Configure handlers
- **Default instance**: `logger` for global use in the project

### `layers/activations/`

- **Unified base**: All activations inherit from `Activation` which in turn inherits from `Layer` and `Module`

#### `activations.py`

- **Class `Activation`**:
  - **Inheritance**: Subclass of `Layer`
  - **Purpose**: Base class for all activation layers
  - **Attributes**:
    - `name`: Lowercase class name (identifier for initialization maps)
    - `affect_init`: Boolean indicating if the activation affects weight initialization
  - **Methods**:
    - `get_init_key()`: Returns the initialization key if `affect_init = True`
    - `init_key`: Property alias of `get_init_key()`

#### `relu.py`

- **Class `ReLU`**:

  - **Inheritance**: Subclass of `Activation`
  - **Configuration**: `affect_init = True`
  - **Attributes**:
    - `_mask`: Boolean mask saved during forward pass
  - **Forward**: Applies ReLU function: $max(0, x)$ element-wise
  - **Backward**: Backpropagates gradient using the mask: `grad * _mask`

- **Class `LeakyReLU`**:
  - **Inheritance**: Subclass of `Activation`
  - **Parameters**:
    - `negative_slope` (default 0.01): Slope for negative values
  - **Attributes**:
    - `a`: Stores the value of `negative_slope`
    - `activation_param`: same value as `a` (for consistency)
    - `_cache_input`: Input saved for backward pass
  - **Forward**: Applies `x if x >= 0, else α * x`
  - **Backward**: Calculates `grad * np.where(x >= 0, 1.0, self.a)`

#### `sigmoid.py`

- **Class `Sigmoid`**:
  - **Inheritance**: Subclass of `Activation`
  - **Configuration**: `affect_init = True`
  - **Attributes**:
    - `out`: Output saved from forward pass for use in backward
  - **Forward**: Calculates sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$
  - **Backward**: Calculates gradient: $\frac{\partial L}{\partial a} \cdot \sigma(x) \cdot (1 - \sigma(x))`

#### `softmax.py`

- **Class `SoftMax`**:
  - **Inheritance**: Subclass of `Activation`
  - **Configuration**: `affect_init = False` (does not affect initialization)
  - **Integration with losses**: Designed to be used with `CrossEntropyLoss` (affect_init=False)
  - **Parameters**:
    - `axis` (default 1): Axis to apply softmax on
  - **Attributes**:
    - `out`: Output saved from forward pass
  - **Forward**: Calculates numerically stable softmax:
    $$S(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
  - **Backward**: Calculates Jacobian-vector product:
    $$\frac{\partial S_i}{\partial z_j} = S_i(\delta_{ij} - S_j)$$

#### `tanh.py`

- **Class `Tanh`**:
  - **Inheritance**: Subclass of `Activation`
  - **Configuration**: `affect_init = True`
  - **Attributes**:
    - `out`: Output saved from forward pass for use in backward
  - **Forward**: Calculates hyperbolic tangent function: $\tanh(x)$
  - **Backward**: Calculates gradient: $\frac{\partial L}{\partial a} \cdot (1 - \tanh^2(x))$

### `layers/bn/`

#### `batch_normalization.py`

- **Class `BatchNormalization`**:

  - **Algorithm**: Batch Normalization (Ioffe & Szegedy, 2015)
  - **Forward Pass Formulas (Training Mode)**:

    **Minibatch statistics**:
    $$\mu = \frac{1}{m} \sum_{i=1}^{m} x_i$$
    $$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2$$

    **Normalization**:
    $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

    **Scale and shift**:
    $$y_i = \gamma \hat{x}_i + \beta$$

    **Moving statistics update**:
    $$\text{running\_mean} = (1 - \text{momentum}) \cdot \text{running\_mean} + \text{momentum} \cdot \mu$$
    $$\text{running\_var} = (1 - \text{momentum}) \cdot \text{running\_var} + \text{momentum} \cdot \sigma^2$$

  - **Backward Pass Formulas**:

    **Gradients with respect to parameters**:
    $$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$
    $$\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}$$

    **Gradient with respect to input** (efficient vectorized version):
    $$\frac{\partial L}{\partial x_i} = \frac{\gamma}{m \sqrt{\sigma^2 + \epsilon}} \left( m \frac{\partial L}{\partial \hat{x}_i} - \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} \hat{x}_j \right)$$

    Where:

    - $\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$
    - $m$ is the minibatch size

  - **Evaluation Mode**:
    $$\hat{x}_i = \frac{x_i - \text{running\_mean}}{\sqrt{\text{running\_var} + \epsilon}}$$
    $$y_i = \gamma \hat{x}_i + \beta$$

  - **Reference**: Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"

## **Implementation Details:**

### **Numerical Stability**:

- **Unbiased variance**: Uses $\frac{m}{m-1}$ for bias correction in training
- **Epsilon**: Small $\epsilon$ term to avoid division by zero

### **Caches for Backward**:

- `x_hat`: Normalized values $\hat{x}_i$
- `mu`, `var`: Minibatch mean and variance
- `x_mu`: Differences $x_i - \mu$

### **Key Properties**:

- **Reduction of Internal Covariate Shift**: Stabilizes input distribution
- **Regularization effect**: Reduces dependency on Dropout
- **Allows higher learning rates**: Faster and more stable training

### `layers/linear/`

#### `linear.py`

- **Class `Linear`**:
  - **Inheritance**: Subclass of `Layer`
  - **Purpose**: Linear (fully connected) layer that performs: $y = xW^T + b$
  - **Parameter management**: Uses `Parameters` for `weight` and `bias`, allowing training by optimizers
  - **Integration with Sequential**: Designed to be used within `Sequential` with automatic initialization
  - **Parameters**:
    - `in_features`: Number of input features
    - `out_features`: Number of output features
    - `bias`: Whether to include bias term (default `True`)
    - `init`: Optional initialization function
  - **Attributes**:
    - `weight`: Weight parameter with shape `(out_features, in_features)`
    - `bias`: Bias parameter with shape `(1, out_features)` (optional)
    - `_cache_input`: Input saved for gradient calculation
  - **Methods**:
    - `reset_parameters(initializer)`: (Re)initializes weights and bias
    - `forward(x)`: Calculates $x W^T + b$
    - `backward(grad)`: Calculates gradients for weights, bias and input
    - `parameters()`: Returns list of trainable parameters

### `layers/regularization/`

#### `dropout.py`

- **Class `Dropout`**:
  - **Inheritance**: Subclass of `Layer`
  - **Purpose**: Regularization by randomly turning off neurons
  - **Parameters**:
    - `p`: Dropout probability (0.0 ≤ p < 1.0)
  - **Attributes**:
    - `_mask`: Binary mask of preserved elements
  - **Behavior**:
    - **Training**: Randomly turns off neurons and scales the remaining ones
    - **Evaluation**: Passes input unchanged
  - **Methods**:
    - `forward(x)`: Applies dropout during training
    - `backward(grad)`: Propagates gradients only through active neurons

### `losses/`

#### `functional.py`

- **Class `CrossEntropyLoss`**:

  - **Purpose**: Cross entropy loss for multiclass classification
  - **Features**:
    - Expects `logits` of shape `(N, C)` and labels as integer indices `(N,)`
    - Uses numerically stable softmax internally
    - Handles both one-hot labels and class indices
  - **Attributes**:
    - `y_hat`: Cached softmax probabilities
    - `y_one_hot`: One-hot encoded labels
    - `eps`: Numerical stability term (1e-12)
  - **Methods**:
    - `forward(logits, targets)`: Calculates loss and caches values
    - `backward()`: Returns gradient with respect to logits: $\frac{\partial L}{\partial z} = \frac{\hat{y} - y}{N}$
    - `__call__()`: Convenience to get loss and gradient together

- **Class `MSE`**:

  - **Purpose**: Mean squared error for regression
  - **Forward**: $\frac{1}{N} \sum (logits - targets)^2$
  - **Backward**: $\frac{2}{N} (logits - targets)$

- **Class `MAE`**:

  - **Purpose**: Mean absolute error for regression
  - **Forward**: $\frac{1}{N} \sum |logits - targets|$
  - **Backward**: $\frac{sign(logits - targets)}{N}$

- **Class `BinaryCrossEntropy`**:
  - **Purpose**: Binary cross entropy for binary classification
  - **Expects**: Probabilities (after sigmoid) and labels 0/1
  - **Forward**: $-\frac{1}{N} \sum [y \cdot \log(p) + (1-y) \cdot \log(1-p)]$
  - **Backward**: $\frac{p - y}{N}$

### `metrics/`

#### `metrics.py`

- **Function `accuracy`**:

  - **Purpose**: Accuracy for multiclass classification
  - **Input**: Model that returns logits/probabilities and DataLoader
  - **Calculation**: $\frac{\text{correct predictions}}{\text{total samples}}$
  - **Usage**: For problems like MNIST, Fashion-MNIST with 10 classes

- **Function `binary_accuracy`**:

  - **Purpose**: Accuracy for binary classification
  - **Input**: Model that returns probabilities [0,1] and DataLoader
  - **Threshold**: 0.5 to convert probabilities to classes 0/1
  - **Usage**: For binary problems like make_moons

- **Function `r2_score`**:
  - **Purpose**: Coefficient of determination for regression
  - **Formula**: $R^2 = 1 - \frac{SSE}{SST}$
  - **Interpretation**:
    - 1.0: perfect fit
    - 0.0: model same as mean
    - < 0.0: model worse than mean

### `model/`

#### `nn.py`

- **Class `Sequential`**:
  - **Inheritance**: Subclass of `Layer`
  - **Purpose**: Sequential container for layers (similar to PyTorch)
  - **Features**:
    - Automatic initialization of linear layers based on adjacent activations
    - Unified management of modes (train/eval) for all layers
    - Automatic parameter collection
  - **Smart initialization**: `Sequential` automatically selects optimal initializers based on activations near each linear layer
  - **Integration with `core/config.py`**: Uses `DEFAULT_NORMAL_INIT_MAP` to get initialization functions
  - **Support for LeakyReLU**: Detects parameter `a` and uses `kaiming_normal_` with the correct slope
  - **Methods**:
    - `forward(x)`: Sequential propagation through all layers
    - `backward(grad)`: Backpropagation in reverse order
    - `parameters()`: All parameters from all layers
    - `train()`/`eval()`: Sets mode for all layers

### `module/`

#### `module.py`

- **Class `Parameters`**:

  - **Purpose**: Container for trainable tensors and their gradients
  - **Attributes**:
    - `data`: Current parameter values
    - `grad`: Accumulated gradients (same shape as `data`)
    - `name`: Optional identifier for debugging
  - **Methods**:
    - `zero_grad(set_to_none)`: Resets gradients to zero or `None`

- **Class `Module`**:
  - **Purpose**: Base class for all network components
  - **Attributes**:
    - `_training`: Flag indicating training/evaluation mode
  - **Methods**:
    - `train()`: Activates training mode
    - `eval()`: Activates evaluation mode
    - `parameters()`: Returns trainable parameters (empty by default)
    - `zero_grad()`: Resets gradients of all parameters

#### `layer.py`

- **Class `Layer`**:
  - **Inheritance**: Subclass of `Module` and `ABC` (abstract class)
  - **Purpose**: Abstract base for all network layers
  - **Abstract methods**:
    - `forward(x)`: Transformation from input to output
    - `backward(grad)`: Gradient calculation with respect to input

### `optim/`

#### `sgd.py`

- **Class `SGD`**:

  - **Algorithm**: Stochastic Gradient Descent with Momentum (Polyak, 1964)
  - **Formulas**:

    **Momentum update**:
    $$v_t = \beta v_{t-1} - \eta \nabla J(\theta_t)$$

    **Parameter update**:
    $$\theta_{t+1} = \theta_t + v_t$$

    **With L2 weight decay**:
    $$\theta_{t+1} = \theta_t + v_t - \eta \lambda \theta_t$$

    **With L1 weight decay**:
    $$\theta_{t+1} = \theta_t + v_t - \eta \lambda \text{sign}(\theta_t)$$

  - **Gradient Clipping** (optional):
    $$\text{If } \|g\| > c, \quad g = \frac{c \cdot g}{\|g\|}$$

#### `rmsprop.py`

- **Class `RMSprop`**:

  - **Algorithm**: RMSprop (Hinton, 2012) - not formally published
  - **Formulas**:

    **Moving average of squared gradients**:
    $$E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2$$

    **Parameter update**:
    $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

#### `adam.py`

- **Class `Adam`**:

  - **Algorithm**: Adam (Kingma & Ba, 2014) - "Adaptive Moment Estimation"
  - **Formulas**:

    **First and second order moments**:
    $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
    $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

    **Bias correction**:
    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
    $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

    **Parameter update**:
    $$\theta_{t+1} = \theta_t - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

## **Common Features of Optimizers:**

### **Weight Decay**:

- **L2 Regularization**: $\theta \leftarrow \theta - \eta \lambda \theta$
- **L1 Regularization**: $\theta \leftarrow \theta - \eta \lambda \text{sign}(\theta)$
- **Exclusion**: BatchNorm `gamma`/`beta` parameters are excluded from weight decay

### **Gradient Clipping** (SGD):

- **Purpose**: Prevent gradient explosion
- **Formula**: $\text{clip}(g, c) = \frac{c \cdot g}{\max(\|g\|, c)}$

### **Initialization**:

- Moments (`velocities`, `moments`) initialized to zero
- Step counter (`t`) in Adam for bias correction

## 📚 **Algorithm References:**

1. **SGD with Momentum**: Polyak, B. T. (1964). "Some methods of speeding up the convergence of iteration methods"
2. **RMSprop**: Hinton, G. (2012). Lecture 6e of "Neural Networks for Machine Learning"
3. **Adam**: Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization"

### `utils.py`

**Purpose**: Utilities for gradient testing and dataset loading

#### **Gradient Testing Functions:**

- **`numeric_grad_elementwise(act_forward, x, eps=1e-6)`**:

  - **Purpose**: Element-wise numerical gradient for vector functions
  - **Method**: Central differences
  - **Formula**:
    $$\frac{\partial f_i}{\partial x_j} \approx \frac{f_i(x_j + \epsilon) - f_i(x_j - \epsilon)}{2\epsilon}$$
  - **Usage**: Verify gradients of activations (ReLU, Sigmoid, Tanh)

- **`numeric_grad_scalar_from_softmax(softmax_forward, x, G, eps=1e-6)`**:

  - **Purpose**: Numerical gradient for softmax + scalar loss
  - **Context**: $L = \sum(\text{softmax}(x) \cdot G)$
  - **Formula**:
    $$\frac{\partial L}{\partial x_j} \approx \frac{L(x_j + \epsilon) - L(x_j - \epsilon)}{2\epsilon}$$
  - **Usage**: Specific testing for SoftMax + CrossEntropy

- **`numeric_grad_scalar_wrt_x(forward_fn, x, G, eps=1e-6)`**:

  - **Purpose**: Generic numerical gradient for scalar losses
  - **Context**: $S = \sum(\text{forward}(x) \cdot G)$
  - **Formula**: Similar to softmax but for any function
  - **Usage**: General layer testing

- **`numeric_grad_wrt_param(layer, param_attr, x, G, eps=1e-6)`**:
  - **Purpose**: Numerical gradient with respect to layer parameters
  - **Context**: $S = \sum(\text{layer.forward}(x) \cdot G)$
  - **Formula**:
    $$\frac{\partial S}{\partial p_j} \approx \frac{S(p_j + \epsilon) - S(p_j - \epsilon)}{2\epsilon}$$
  - **Usage**: Verify gradients of weights and biases in linear layers

#### **Data Loading Functions:**

- **`normalize(x_data, x_mean, x_std)`**:

  - **Purpose**: Standard data normalization
  - **Formula**:
    $$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

- **`_split_features_and_labels(df)`**:

  - **Purpose**: Separate features and labels from DataFrames
  - **Logic**:
    - If "label" column exists → use it
    - If not → first column are labels

- **`load_fashion_mnist_data(...)`**:

  - **Purpose**: Load Fashion-MNIST dataset
  - **Features**:
    - Optional normalization with training set statistics
    - Uses PyArrow for memory efficiency
    - Automatic split into train/val/test

- **`load_mnist_data(...)`**:
  - **Purpose**: Load classic MNIST dataset
  - **Same features** as Fashion-MNIST

## 🏗️ Framework Architecture

The framework follows a modular architecture inspired by PyTorch with a well-defined inheritance system:

### **Main Inheritance System**

```
Module (base)
    ↳ Layer (abstracta)
        ↳ Linear, Activation, BatchNormalization, Dropout
            ↳ Activation (base para activaciones)
                ↳ ReLU, LeakyReLU, Sigmoid, SoftMax, Tanh
```

### **Data Flow and Responsibilities**

- **`Module`**: State management (train/eval) and trainable parameters
- **`Layer`**: Abstract interface for transformations (forward/backward)
- **`Parameters`**: Container for parameter data and gradients
- **Concrete Layers**: Specific implementations of transformations
- **`Losses`**: Loss functions with the same forward/backward pattern

### **Smart Initialization System**

The framework includes automatic parameter initialization:

```python
# Sequential automatically detects activations and chooses optimal initializers
model = Sequential(
    Linear(784, 256),  # Initialized with Kaiming (due to following ReLU)
    ReLU(),
    Linear(256, 10)    # Initialized with Kaiming (due to previous ReLU)
)
```

### **Optimization System**

The framework implements modern optimizers with their original formulas:

- **SGD with Momentum** (Polyak, 1964): Acceleration by inertia
- **RMSprop** (Hinton, 2012): Per-parameter adaptive learning rate
- **Adam** (Kingma & Ba, 2014): Combination of momentum and adaptation

**Advanced features**:

- Weight decay (L1/L2) with automatic BatchNorm exclusion
- Gradient clipping for numerical stability
- Bias correction in Adam (important in early iterations)

### **Batch Normalization (BatchNorm)**

The framework implements Batch Normalization according to the original paper by Ioffe and Szegedy (2015):

- **Training Stabilization**: Reduces internal covariate shift
- **Training Acceleration**: Allows using higher learning rates
- **Light Regularization**: Reduces the need for Dropout

**Advanced features**:

- **Different modes**: Different behavior in training vs evaluation
- **Moving statistics**: Accumulated during training for evaluation
- **Learnable parameters**: `gamma` (scale) and `beta` (shift)
- **Smart exclusion**: Optimizers exclude `gamma`/`beta` from weight decay

### **Unified Training Pattern**

Based on the real examples in the project, the typical workflow is:

```python
# 1. MODEL CONFIGURATION
model = Sequential(
    Linear(784, 256),
    BatchNormalization(256),
    ReLU(),
    Dropout(0.3),
    Linear(256, 10),
)

# 2. DATA PREPARATION
train_loader = DataLoader(x_train, y_train, batch_size=256, shuffle=True)
val_loader = DataLoader(x_val, y_val, batch_size=256, shuffle=False)

# 3. TRAINING CONFIGURATION
model.train()  # Training mode
optimizer = Adam(model.parameters(), learning_rate=1e-3, weight_decay=1e-4)
loss_fn = CrossEntropyLoss()

# 4. TRAINING LOOP
for epoch in range(epochs):
    for input, target in train_loader:
        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        output = model.forward(input)

        # Calculate loss
        loss, grad = loss_fn(output, target)

        # Backward pass
        model.backward(grad)

        # Update parameters
        optimizer.step()

    # Periodic validation
    model.eval()
    accuracy = accuracy(model, val_loader)
    model.train()
```

### **Specific Patterns by Problem Type**

**Multiclass Classification (MNIST, Fashion-MNIST)**

```python
# Typical Architecture
Sequential(
Linear(784, 512), BatchNormalization(512), ReLU(), Dropout(0.3),
Linear(512, 256), BatchNormalization(256), ReLU(), Dropout(0.3),
Linear(256, 10) # No activation - logits for CrossEntropy
)
loss_fn = CrossEntropyLoss() # Includes SoftMax internally
```

**Binary Classification (Make Moons)**

```python
# Typical Architecture
Sequential(
Linear(2, 32), Tanh(), Dropout(0.2),
Linear(32, 1), Sigmoid() # Probabilities for BinaryCrossEntropy
)
loss_fn = BinaryCrossEntropy()
```

**Regression (Make Regression)**

```python
# Typical Architecture
Sequential(
Linear(45, 368), BatchNormalization(368), LeakyReLU(), Dropout(0.2),
Linear(368, 176), BatchNormalization(176), LeakyReLU(), Dropout(0.4),
Linear(176, 1) # Continuous Output
)
loss_fn = MSE() # or MAE()
```

### **Evaluation Components**

```python
# Metrics by Problem Type
accuracy(model, data_loader) # Multiclass Classification
binary_accuracy(model, data_loader) # Binary Classification
r2_score(model, data_loader) # Regression
```

## 🛠️ Technologies Used

- Languages: Python 3.14.0 🐍

- Development Tools: Extensions: `Black Formatter`, `FileTree Pro`

- Main Libraries Used:

1. **`numpy`**: Arrays with memory-optimized vector operations.
2. **`pandas`**: Handling tabular data. E.g., datasets like _fashion-mnist_ and _mnist_.
3. **`matplotlib`**: Data visualization and statistical graphs.
4. **`seaborn`**: Improved styles for visualizations.
5. **`scikit-learn`**: Classical Machine Learning library.
6. **`pyarrow`**: Reduces pandas' high memory usage with large datasets.
7. **`pytest`**: Unit tests.

## 📦 Installation

Instructions for installing dependencies and preparing the environment

1. **Clone repository**

```bash
git clone https://github.com/JOSE-MDG/NovaNN.git

# Access the directory
cd NovaNN
```

2. **Create and activate a virtual environment**

- Windows:

```bash
python -m venv .venv

# Activate virtual environment
.\\.venv\\Scripts\\activate

# If the above causes problems (PowerShell)
.\\.venv\\Scripts\\Activate.Ps1
```

- Linux/MacOS:

```bash
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

3. **Install the requirements (`requirements.txt `)**

```bash
pip install -r requirements.txt
```

## 🧪 Testing (`tests/`)

The framework includes a complete suite of unit tests that verify the correct implementation of all components:

### **Activation Tests** (`tests/activations/`)

#### `test_relu.py`

- **Verifies**: Forward pass (non-negativity) and backward pass (gradient mask)
- **Properties**: Output shape, behavior at x>0 and x<0
- **Gradients**: Analytical vs numerical comparison for x ≠ 0

#### `test_leaky_relu.py`

- **Verifies**: Behavior with negative slope (`negative_slope`)
- **Properties**: Forward (x≥0 passes, x<0 scales) and backward (1.0 vs slope)
- **Gradients**: Numerical validation for non-zero inputs

#### `test_sigmoid.py`

- **Verifies**: Output range (0,1) and derivative calculation
- **Properties**: Output shape and forward/backward consistency
- **Gradients**: Comparison σ(x)\*(1-σ(x)) vs numerical

#### `test_softmax.py`

- **Verifies**: Numerical stability and probability properties
- **Properties**: Sums to 1 per row, shift invariance
- **Jacobian**: Jacobian-vector product vs numerical approximation

#### `test_tanh.py`

- **Verifies**: Output range (-1,1) and odd function property
- **Properties**: tanh(-x) = -tanh(x), derivative 1 - tanh²(x)
- **Gradients**: Numerical validation of derivative

### **BatchNorm Tests** (`tests/batch_norm/`)

#### `test_batch_norm.py`

- **Verifies**: Behavior in training/evaluation modes
- **Training**: Correct normalization (mean ~0, bounded variance)
- **Evaluation**: Use of moving statistics, stable outputs
- **Properties**: Shape preservation, feature centering

### **DataLoader Tests** (`tests/dataloader/`)

#### `test_dataloader.py`

- **Verifies**: Correct batch handling, including smaller last batch
- **Properties**: Shape preservation, complete dataset iteration
- **Edge cases**: Batch_size that doesn't exactly divide dataset

### **Initializer Tests** (`tests/initializers/`)

#### `test_init.py`

- **Verifies**: Statistical distributions of initializers
- **Kaiming**: Zero mean, appropriate variance for ReLU/LeakyReLU
- **Xavier**: Correct scaling for tanh/sigmoid
- **Validation**: Uniform distribution limits, error handling
- **Exceptions**: Unsupported nonlinearities raise ValueError

### **Regularization Tests**

#### `test_dropout_regularization.py`

- **Verifies**: Behavior in training vs evaluation modes
- **Training**: Application of binary mask and correct scaling (1/(1-p))
- **Evaluation**: Pass-through unchanged
- **Backward**: Gradients masked and scaled identically to forward

### **Linear Layer Tests**

#### `test_linear_layer.py`

- **Verifies**: Forward/backward of linear layers with and without bias
- **Shapes**: Preservation of input/output dimensions
- **Gradients**: Numerical validation of gradients with respect to inputs and parameters
- **Parameters**: Weight and bias gradients vs numerical approximations

### **Sequential Module Tests**

#### `test_sequential_module.py`

- **Verifies**: Behavior of Sequential containers
- **Forward/Backward**: Propagation through multiple layers
- **Initialization**: Automatic detection of activations for initializers
- **Parameters**: Correct collection of all trainable parameters
- **Shapes**: Dimension preservation through the network

### **Testing Methodology**

- **Numerical gradients**: Using functions from `utils.py` to verify backpropagation
- **Analytical comparison**: `assert_allclose` with tolerance 1e-5
- **Training/eval modes**: Verification of different behaviors
- **Complete coverage**: All trainable parameters and edge cases
- **Deterministic RNG**: For reproducible tests

### **Test Execution**

```bash
# Run all tests
pytest tests/

# Run tests with verbose output
pytest tests/ -v

# Run specific tests
pytest tests/activations/test_relu.py
pytest tests/batch_norm/ -v
```

## 🤝 Contribution

Contributions are welcome and appreciated. If you wish to contribute to NovaNN Framework, please follow these steps:

### **How to Contribute?**

1. **Fork the project**
2. **Create a branch for your feature** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### **Contribution Areas**

- 🐛 **Bug reports** and issues
- 💡 **New features** and improvements
- 📚 **Documentation improvements**
- 🧪 **Additional tests**

### **Style Guides**

- Follow existing code conventions
- Include tests for new functionality
- Update corresponding documentation

### **Issue Reporting**

When reporting a bug, please include:

- Python version and dependencies
- Steps to reproduce the problem
- Expected vs current behavior
- Relevant logs if applicable

## 📄 License

This project is under the **MIT License**. See the [LICENSE](LICENCE) file for more details.

## 👤 Author

Juan José - Developer, Machine & Deep Learning Enthusiast.
GitHub: https://github.com/JOSE-MDG

```

```
