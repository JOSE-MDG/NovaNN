![Banner](./images/NovaNN%20Oficial%20Banner.jpg)

![version](https://img.shields.io/badge/version-4.0.4-blue)
![python](https://img.shields.io/badge/python-v3.14-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![tests](https://img.shields.io/badge/tests-pytest-orange)
![coverage](https://img.shields.io/badge/coverage-87%25-success)
![v5](https://img.shields.io/badge/v5.0.0-active%20development-orange)

## Available Languages

- [English](README.md)
- [Español](README.es.md)

## Table of Contents

- [What is NovaNN?](#what-is-novann)
- [Project Philosophy](#project-philosophy)
- [Numerical Backend](#numerical-backend)
- [Educational and Technical Goal](#educational-and-technical-goal)
- [Introduction](#introduction)
  - [Project Organization](#project-organization)
- [Quick Start](#quick-start)
  - [Autograd and Computational Graphs](#1-autograd-and-computational-graphs)
  - [Complete Neural Network Training](#2-complete-neural-network-training)
  - [CNN Architectures for Computer Vision](#3-cnn-architectures-for-computer-vision)
  - [Transfer Learning - Freezing Layers](#4-transfer-learning---freezing-layers)
  - [Fine-Tuning with Differential Learning Rates](#5-fine-tuning-with-differential-learning-rates)
  - [Model Saving and Loading](#6-model-saving-and-loading)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
  - [Install from PyPI (Recommended)](#option-1-install-from-pypi-recommended)
  - [Install from source](#option-2-install-from-source)
  - [Run examples](#run-examples)
  - [Uninstallation](#uninstallation)
  - [Troubleshooting](#troubleshooting)
- [Testing](#testing)
  - [Run All Tests](#run-all-tests)
- [Contribution](#contribution)
- [License](#license)
- [Author and Maintainer](#author-and-maintainer)

## What is NovaNN?

**NovaNN** is a **[Deep Learning](https://www.ibm.com/think/topics/deep-learning)** framework developed from scratch in **Python**, designed to build, train, and evaluate neural networks in a modular, clear, and extensible way.

The main goal of NovaNN is not to compete with industrial frameworks, but to **understand, implement, and demonstrate** how modern frameworks like **[PyTorch](https://docs.pytorch.org/docs/stable/index.html)** or **[TensorFlow](https://www.tensorflow.org/api_docs)** work internally, with special emphasis on the architecture of **PyTorch**, which served as the main inspiration.

NovaNN allows defining complete neural models, managing training, and performing automatic backpropagation through a **dynamic autograd engine**, all built explicitly without relying on external computation engines.

## Project Philosophy

NovaNN was born with a clear idea:

> _Don't use the magic of existing frameworks, but build it._

Every component of the framework is designed to be **readable, traceable, and testable**, prioritizing deep understanding of:

- How computational graphs are built
- How gradients flow during backward
- How scalable ML frameworks are structured
- How clean and extensible APIs are designed

## Numerical Backend

NovaNN uses **NumPy** as the main backend for numerical computation, leveraging:

- Efficient vectorized operations
- Explicit tensor manipulation
- Complete control over mathematical operations

This allows focusing on the **logic of Deep Learning** (autograd, layers, optimization, training) without overly abstracting the internal system behavior.

## Educational and Technical Goal

This framework was created for **educational and demonstrative purposes**, with the aim of demonstrating solid knowledge in:

- **Machine Learning and Deep Learning**
- **Mathematical fundamentals** (linear algebra, calculus, optimization)
- **Autograd and backpropagation**
- **System design and software architecture**
- **Unit testing and numerical validation**
- **Modular and extensible design**
- **Software engineering best practices**
- **Data preprocessing and model training**

NovaNN is intended for people who want to **understand how Deep Learning frameworks really work inside**, beyond just using them.

> ⚠️ **Note**
> NovaNN does not aim to replace frameworks like PyTorch or TensorFlow in production environments.
> Its purpose is to serve as an advanced learning tool and as a technical demonstration of engineering applied to Deep Learning.

## Introduction

**NovaNN** adopts a **modular organization** inspired by modern Deep Learning frameworks, with responsibilities clearly separated between data, models, training, and utilities.
This structure favors both extensibility and clarity of workflow.

### Project Organization

- **`examples/`**
  Contains functional scripts showing the use of the framework in different scenarios:
  - Binary classification
  - Multiclass classification
  - Regression
  - Convolutional networks

- **[`nova/`](./nova/README.md)**
  Contains the **complete core of the NovaNN framework**.
  Here tensors, the autograd engine, mathematical operations, neural network modules, optimizers, metrics, serialization, and internal utilities are implemented.
  It is organized in a modular way to clearly separate the different levels of the system: low-level (tensors and operations), autograd, high-level APIs (`nn`, `optim`, `metrics`) and auxiliary utilities.
  Each submodule has its own documentation to facilitate code navigation and maintenance.

- **[`benchmarks/`](./benchmarks/README.md)**
  Includes **benchmarks designed to evaluate NovaNN's performance** in different scenarios and compare it with other frameworks (mainly PyTorch).
  The benchmarks focus on:
  - Elementary operations and reduction
  - Autograd system cost
  - CPU training on small models
  - Memory usage and computational overhead
    This directory is not part of the framework runtime and is intended exclusively for **performance analysis, technical validation, and comparative studies**.

## Quick Start

Build and train models with a syntax you already know. NovaNN looks like PyTorch, but runs on your own custom engine.

### 1. **Autograd and Computational Graphs**

Experiment with NovaNN's automatic differentiation engine. Create tensors, perform operations, and observe how gradients flow.

```python
import nova
import nova.nn as nn

# Create tensors with gradient tracking
x = nova.tensor([[0.5, -0.2]], requires_grad=True)
w = nova.tensor([[1.0], [0.5]], requires_grad=True)
b = nova.tensor([0.1], requires_grad=True)

# Manual forward pass
y = x @ w + b
loss = (y ** 2).sum()

# Automatic backward pass
loss.backward()

print(f"Loss: {loss.item()}")           # Loss: 0.25
print(f"Gradient of x: {x.grad}")     # Automatically calculated gradients
print(f"Gradient of w: {w.grad}")
print(f"Gradient of b: {b.grad}")

# Or using nn.Module layers
model = nn.Linear(2, 1)
output = model(x)
output.backward()
```

### 2. **Complete Neural Network Training**

Train a simple binary classifier with all the framework's functionalities.

```python
import nova
import nova.nn as nn
import nova.optim as optim
from nova.nn import functional as F

# Define the model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

# Create model and optimizer
model = BinaryClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Example data
X_train = nova.randn(100, 10)  # 100 samples, 10 features
y_train = nova.randint(0, 2, (100, 1))

# Training loop
for epoch in range(50):
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
```

### 3. **CNN Architectures for Computer Vision**

NovaNN supports complex modules like 2D convolutions, batch normalization, and lazy layers.

```python
import nova
import nova.nn as nn
import nova.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.1)

        # Convolutional block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)

        # Convolutional block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers (lazy to automatically infer dimensions)
        self.fc1 = nn.LazyLinear(256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4

        # Flatten
        x = x.view(x.size(0), -1)

        # Classification head
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Create model and process images
model = ConvNet(num_classes=10)
batch_images = nova.rand(8, 3, 32, 32)  # Batch of 8 RGB 32x32 images
logits = model(batch_images)

print(f"Output: {logits.shape}")  # Shape: (8, 10)
```

### 4. **Transfer Learning - Freezing Layers**

Leverage pre-trained models by freezing layers to use them as fixed feature extractors.

```python
import nova
import nova.nn as nn
import nova.optim as optim
from nova.nn import functional as F

# Suppose we have a pre-trained model
class PretrainedFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

# Complete model with transfer learning
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Pre-trained backbone (frozen)
        self.backbone = PretrainedFeatureExtractor()

        # New classification head (trainable)
        self.classifier = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create model
model = TransferLearningModel(num_classes=5)

# Freeze feature extractor layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Only train the classifier
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# Verify which parameters are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable}/{total}")
```

### 5. **Fine-Tuning with Differential Learning Rates**

Train different parts of the network with different learning rates for optimal fine-tuning.

```python
import nova
import nova.nn as nn
import nova.optim as optim
from nova.nn import functional as F

class FineTuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Base layers (pre-trained)
        self.base_layers = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Middle layers
        self.mid_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Classification head (new)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        x = self.base_layers(x)
        x = self.mid_layers(x)
        x = self.head(x)
        return x

model = FineTuneModel()

# Configure differential learning rates per layer group
optimizer = optim.Adam([
    {'params': list(model.base_layers.parameters()), 'lr': 1e-5},  # Very low for base layers
    {'params': list(model.mid_layers.parameters()), 'lr': 1e-4},   # Medium
    {'params': list(model.head.parameters())}
], lr=1e-3) # High for new head

# Example data
X = nova.randn(32, 100)
y = nova.randint(0, 10, (32,))

# Training
for epoch in range(100):
    logits = model(X)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

### 6. **Model Saving and Loading**

Serialize your trained models to reuse them later.

```python
import nova
import nova.nn as nn

# Train model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Save complete model
nova.save(model, 'model.pt')

# Save only parameters (state_dict)
nova.save(model.state_dict(), 'model_weights.pt')

# Load complete model
loaded_model = nova.load('model.pt')

# Load only parameters into a new model
new_model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
new_model.load_state_dict(nova.load('model_weights.pt'))
```

## Technologies Used

The **NovaNN** framework is built using the following main technologies and libraries:

- **Language**: Python >= 3.14, < 3.15
- **Dependency Management**: Poetry (for package management and virtual environments)
- **Main Libraries**:
  - `numpy`: Efficient numerical operations and multidimensional arrays
  - `pandas`: Tabular data handling and analysis (for dataset loading)
  - `matplotlib`: Graph and result visualization
  - `seaborn`: Aesthetic enhancement of statistical visualizations
  - `scikit-learn`: Classical Machine Learning tools and utilities
  - `pyarrow`: Efficient backend for pandas DataFrames (reduces memory usage)
  - `pyyaml`: to manipulate YAML files
  - `requests`: To make web queries
  - `tqdm`: To show progress bar
- **Development Tools**:
  - `pytest`: Unit testing framework
  - `pytest-cov`: Code coverage in tests
  - `ipykernel`: Jupyter kernel for notebooks
  - `black`: Code formatter to maintain consistent style
- **Benchmarking Tools**
  - `torch`: Deep learning framework
  - `torchvision`: Extra torch package for vision tasks

## Installation

NovaNN is available on **PyPI** and can be easily installed using `pip` or `poetry`. You can also install it from source if you want to contribute or explore the framework in depth.

### Option 1: Install from PyPI (Recommended)

The easiest way to install NovaNN is via pip or poetry:

```bash
# With pip
pip install novann

# With poetry
poetry add novann
```

#### Verify installation

```python
import nova
print(nova.__version__)  # Should display: 4.0.4
```

#### System requirements

- **Python**: >= 3.12, < 4.0.0
- **Operating system**: Windows, Linux, macOS

### Option 2: Install from source

If you want to contribute to the project or explore the source code, you can clone the repository and install using Poetry.

#### 1. Clone the repository

```bash
git clone git@github.com:JOSE-MDG/NovaNN.git
cd NovaNN
```

#### 2. Install Poetry (if you don't have it)

- Windows (PowerShell):

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

- Linux/macOS:

```bash
# With curl
curl -sSL https://install.python-poetry.org | python3 -

# With pipx
pipx install poetry
```

#### Add Poetry to PATH:

- On Linux/macOS:

```bash
# Bash/Zsh (temporary)
export PATH="$HOME/.local/bin:$PATH"

# Bash (permanent - add to ~/.bashrc)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Zsh (permanent - add to ~/.zshrc)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

- On Windows

```powershell
# PowerShell (temporary for current session)
$env:Path += ";$env:APPDATA\Python\Scripts"

# PowerShell (permanent - current user)
[System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:APPDATA\Python\Scripts", "User")

# PowerShell (permanent - system)
[System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:APPDATA\Python\Scripts", "Machine")
```

```cmd
# Command Prompt (temporary)
set PATH=%PATH%;%APPDATA%\Python\Scripts

# Command Prompt (permanent)
setx PATH "%PATH%;%APPDATA%\Python\Scripts"
```

#### 3. Install project dependencies

```bash
# Write lock file
poetry lock

# Install all dependencies (including development)
poetry install

# Production dependencies only
poetry install --without dev,benchmark

# With benchmarking tools
poetry install --with benchmark
```

#### 4. Activate virtual environment

```bash
# Install shell plugin
poetry self add poetry-plugin-shell

# Activate shell with virtual environment
poetry shell

# Alternatively, run commands directly without activating shell:
poetry run python examples/binary_classification.py
```

### Run examples

Once NovaNN is installed (from PyPI or source), you can run the included examples:

```bash
# If installed from source
poetry run python examples/binary_classification.py
poetry run python examples/multiclass_classification.py
poetry run python examples/conv_example.py
poetry run python examples/regression.py

# If installed from PyPI, create your own scripts
python my_neural_network.py
```

### Uninstallation

```bash
# If installed from PyPI with pip
pip uninstall novann

# If installed with poetry
poetry remove novann

# If installed from source
# Simply remove Poetry's virtual environment
poetry env remove python
```

### Troubleshooting

#### Error: "Python version not compatible"

NovaNN requires Python `>=3.12, <4.0.0` Check your version:

```bash
python --version
```

If you have multiple Python versions, use:

```bash
poetry env use python3.14 # or python3.12/13
```

#### Error: "Module nova not found"

If installed from source, make sure you're in Poetry's virtual environment:

```bash
poetry shell
```

Or use `poetry run` before your commands.

#### Problems with NumPy or dependencies

If you have dependency conflicts, try:

```bash
pip install --upgrade pip
pip install novann --no-cache-dir
```

## Testing

The framework includes a complete unit test suite in the [`tests/`](./tests/) directory that verifies correct implementation of all components covering **87%** of the module. For more information go to [Unit Tests](./tests/README.md)

### Run All Tests

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

## Contribution

To know how to contribute to **NovaNN** go to [contributions](./CONTRIBUTING.md)

## License

This project is under the **MIT License**. See the [LICENCE](./LICENCE) file for more details.

**MIT License Summary:**

- Free software to use, copy, modify, merge, publish, distribute
- Can be used for commercial purposes
- License includes original copyright
- No warranty and authors are not responsible for damages

## Author and Maintainer

**Juan José** - Developer & Machine Learning Enthusiast (16 years)

- GitHub: [https://github.com/JOSE-MDG](https://github.com/JOSE-MDG)
- Email: josepemlengineer@gmail.com

**About Me**: At only 16 years old, I built **NovaNN** from scratch as an educational project to demonstrate my passion and deep understanding of deep learning. This framework represents months of self-study, experimentation, and dedication, implementing each algorithm mathematically from original papers.

**Acknowledgments:**

- Inspired by PyTorch and other deep learning frameworks
- Open source community for shared tools and knowledge
- Research papers that form the basis of implementations
