![Banner](./images/NovaNN%20Banners.png)

![version](https://img.shields.io/badge/version-4.0.0-blue)
![python](https://img.shields.io/badge/python-v3.14-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![tests](https://img.shields.io/badge/tests-pytest-orange)
![coverage](https://img.shields.io/badge/coverage-87%25-success)

## 🌐 Available Languages

- 🇬🇧 [English](README.md)
- 🇪🇸 [Español](README.es.md)

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

## 🛠️ Technologies Used

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

## 📦 Installation

NovaNN uses **Poetry** for dependency management and packaging. Follow these steps to set up the environment:

### 1. Clone the Repository

```bash
git clone git@github.com:JOSE-MDG/NovaNN.git
cd NovaNN
```

### 2. Install Poetry (if not installed)

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

- Linux/macOS:

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

- Windows

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

### 3. Add Project to Python Path

- Linux/macOS

```bash
# Temporary
export PYTHONPATH="/path/to/your/project:$PYTHONPATH"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export PYTHONPATH="/path/to/your/project:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

- Windows:

```powershell
# PowerShell (temporary)
$env:PYTHONPATH = "C:\path\to\your\project"

# PowerShell (permanent)
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", "C:\path\to\your\project", "User")
```

```cmd
# Command Prompt (temporary)
set PYTHONPATH=C:\path\to\your\project

# Command Prompt (permanent)
setx PYTHONPATH "C:\path\to\your\project"
```

### 4. Install Project Dependencies

```bash
# Write lock file
poetry lock

# Install all dependencies (including development ones)
poetry install
```

### 5. Activate Virtual Environment

```bash
# Install shell plugin
poetry self add poetry-plugin-shell

# Activate shell with virtual environment
poetry shell

# Alternatively, run commands directly without activating shell:
poetry run python examples/binary_classification.py
```

### 6. Run Examples

```bash
# Binary classification
poetry run python examples/binary_classification.py

# Multiclass classification
poetry run python examples/multiclass_classification.py

# Convolutional networks
poetry run python examples/conv_example.py

# Regression
poetry run python examples/regresion.py
```

## 🧪 Testing

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

## 🤝 Contribution

To know how to contribute to **NovaNN** go to [contributions](./CONTRIBUTING.md)

## 📄 License

This project is under the **MIT License**. See the [LICENCE](./LICENCE) file for more details.

**MIT License Summary:**

- Free software to use, copy, modify, merge, publish, distribute
- Can be used for commercial purposes
- License includes original copyright
- No warranty and authors are not responsible for damages

## 👤 Author and Maintainer

**Juan José** - Developer & Machine Learning Enthusiast (16 years)

- GitHub: [https://github.com/JOSE-MDG](https://github.com/JOSE-MDG)
- Email: josepemlengineer@gmail.com

**About Me**: At only 16 years old, I built **NovaNN** from scratch as an educational project to demonstrate my passion and deep understanding of deep learning. This framework represents months of self-study, experimentation, and dedication, implementing each algorithm mathematically from original papers.

**Acknowledgments:**

- Inspired by PyTorch and other deep learning frameworks
- Open source community for shared tools and knowledge
- Research papers that form the basis of implementations
