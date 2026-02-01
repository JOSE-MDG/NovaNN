# Contribution Guide

Thank you for your interest in contributing to **NovaNN**! This educational open-source project greatly benefits from community contributions.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Proposing Features](#proposing-features)

## Code of Conduct

This project aims to be a welcoming and educational space. It is expected:

- **Mutual respect**: Treat everyone with courtesy and professionalism
- **Constructive criticism**: Focus on the code, not on people
- **Collaboration**: Help others learn and grow
- **Patience**: Remember that we are all learning

## How Can I Contribute?

### Priority Areas

- 🐛 **Bugs**: Report or fix found errors
- 💡 **Features**: New layers, optimizers, or functionalities
- 📚 **Documentation**: Improve READMEs, docstrings, tutorials
- 🧪 **Tests**: Increase coverage and edge cases
- ⚡ **Performance**: NumPy code optimizations
- 🎓 **Tutorials**: Educational examples and usage guides

### General Process

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create a branch** for your change: `git checkout -b feat/new-feature`
4. **Make your changes** following project standards
5. **Commit** with descriptive messages: `feat(nn): add GroupNorm layer`
6. **Push** to your fork: `git push origin feat/new-feature`
7. **Open a Pull Request** in the main repository

## Development Environment Setup

### 1. Prerequisites

- Python >= 3.14, < 3.15
- Poetry (dependency manager)
- Git

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/JOSE-MDG/NovaNN.git
cd NovaNN

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell

# Add project to PYTHONPATH (Linux/macOS)
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Add project to PYTHONPATH (Windows PowerShell)
$env:PYTHONPATH = "$(pwd);$env:PYTHONPATH"
```

[More details](README.md#📦-installation)

### 3. Configure Environment Variables

Create a `.env` file in the root with necessary paths (see `README.md` for details).

### 4. Verify Installation

```bash
# Run tests
poetry run pytest tests/ -v

# Verify coverage
poetry run pytest --cov --cov-report=html
```

## Project Structure

[Complete directory structure](Tree.md)

```
NovaNN/
├── nova/              # Main source code
│   ├── autograd/      # Automatic differentiation system
│   ├── nn/            # Neural network modules
│   ├── optim/         # Optimizers and schedulers
│   ├── metrics/       # Evaluation metrics
│   └── ...
├── tests/             # Unit tests
├── examples/          # Example scripts
└── benchmarks/        # Performance benchmarks
```

### Where Does Each Thing Go?

- **New layer**: `nova/nn/modules/`
- **New optimizer**: `nova/optim/`
- **New metric**: `nova/metrics/`
- **New autograd operation**: `nova/autograd/_ops/` (And define how it's incorporated in the [YAML](nova/autograd/_ops/native/native_functions.yaml))
- **Tests**: `tests/` (mirroring `nova/` structure)
- **Examples**: `examples/` (standalone scripts)
- **Tutorials**: `tutorials/` (commented educational code)

## Code Standards

### Code Style

- **Formatter**: We use **Black** with default configuration

```bash
poetry run black nova/ tests/
```

- **Conventions**: Follow PEP 8 and existing project style
- **Type hints**: Use type annotations consistently

```python
def forward(self, input: Tensor) -> Tensor:
    ...
```

### Naming Conventions

- **Classes**: `PascalCase` (`Linear`, `ReLU`, `SGD`)
- **Functions/methods**: `snake_case` (`forward`, `zero_grad`)
- **Constants**: `UPPER_SNAKE_CASE` (`LOG_FILE`, `MNIST_PATH`)
- **Private**: `_` prefix (`_step_impl`, `_calculate_fans`)

### Docstrings

Use Google/NumPy style docstrings with description, Args, Returns, Examples:

```python
def kaiming_normal_(
    tensor: Parameter,
    a: float = 0.0,
    nonlinearity: str = "leaky_relu"
) -> None:
    """
    Initialize tensor using Kaiming normal initialization.

    Args:
        tensor: Parameter to initialize.
        a: Negative slope for leaky ReLU.
        nonlinearity: Activation function name.

    Examples:
        >>> weight = Parameter(nova.empty((64, 128)))
        >>> init.kaiming_normal_(weight, nonlinearity='relu')
    """
    ...
```

### Commit Messages

We follow **Conventional Commits**:

(commit message format 1)

```text
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

- `feat`: New functionality
- `fix`: Bug fix
- `docs`: Documentation-only changes
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Refactoring without changing functionality
- `test`: Add or fix tests
- `perf`: Performance improvement
- `chore`: Build changes, dependencies, etc.

**Examples:**

```text
feat(nn): add GroupNorm layer
fix(optim): correct AdamW weight decay calculation
docs(tutorials): add transformer example
test(autograd): increase coverage for backward ops
```

## Testing

### Writing Tests

- Use **pytest** for all tests
- One test file per module: `test_<module_name>.py`
- Group related tests in classes: `TestLinear`, `TestSGD`
- Descriptive names: `test_forward_with_bias`, `test_backward_without_grad`

**Example:**

```python
import pytest
import nova
import nova.nn as nn

class TestLinear:
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        layer = nn.Linear(10, 5)
        x = nova.randn(3, 10)
        output = layer(x)
        assert output.shape == (3, 5)

    def test_backward_updates_grad(self):
        """Test that backward pass computes gradients."""
        layer = nn.Linear(10, 5)
        x = nova.randn(3, 10)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        assert layer.weight.grad is not None
```

### Running Tests

```bash
# All tests
poetry run pytest

# Specific tests
poetry run pytest tests/nn/test_linear.py -v

# Specific class
poetry run pytest tests/nn/test_linear.py::TestLinear -v

# Specific test
poetry run pytest tests/nn/test_linear.py::TestLinear::test_forward_shape -v

# With coverage
poetry run pytest --cov --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Minimum Coverage

- Maintain coverage **>= 85%** in new code
- Excluded files: `__init__.py`, `.pyi`, `_internal/`, `_typing/`, `examples/`, `benchmarks/`

## Pull Request Process

### Before Opening the PR

- [ ] Tests pass: `poetry run pytest`
- [ ] Code formatted: `poetry run black nova/ tests/`
- [ ] Coverage maintained or improved
- [ ] Documentation updated (docstrings, READMEs)
- [ ] Commits follow Conventional Commits

### PR Template

```markdown
## Description

[Clear description of changes]

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted with Black
- [ ] Tests pass locally

## Testing

[Describe how you tested the changes]

## Additional Notes

[Any relevant context]
```

### Review Process

1. The maintainer will review your PR within 1-3 days
2. Changes or clarifications may be requested
3. Once approved, it will be merged into `main`
4. Your contribution will appear in the next release

## Reporting Bugs

### Before Reporting

- Search [existing Issues](https://github.com/JOSE-MDG/NovaNN/issues) to avoid duplicates
- Make sure it's a NovaNN bug, not an environment issue

### Issue Template

````markdown
**Bug Description**
[Clear and concise description]

**Steps to Reproduce**

1. Code used
2. Command executed
3. Error observed

**Expected Behavior**
[What you expected to happen]

**Current Behavior**
[What actually happened]

**Environment**

- Python version: [e.g. 3.14.0]
- NovaNN version: [e.g. 4.0.0]
- OS: [e.g. Ubuntu 22.04]
- NumPy version: [e.g. 1.26.0]

**Minimal Reproducible Code**

```python
import nova
# Your code here
```

**Logs/Traceback**
````

[Paste complete error here]

```
**Additional Context**
[Any relevant information]
```

## Proposing Features

### Feature Request Template

```markdown
**Does the feature solve a problem?**
[Describe the problem you face]

**Describe Proposed Solution**
[How you would like it to work]

**Alternatives Considered**
[Other solutions you considered]

**Suggested Implementation**
[If you have ideas on how to implement it]

**Additional Context**
[Papers, references, examples from other frameworks]
```

## Possible Questions

### Can I contribute as a beginner?

Absolutely! Issues labeled as `good first issue` are ideal to start.

### In which language should I write code/docs?

- **Code**: English (variable names, functions, comments)
- **Documentation**: Spanish and English (both welcome)
- **Issues/PRs**: Spanish or English

### Do I need to write tests for docs?

Not necessary for documentation-only changes, but yes for new code.

### How long does the review take?

Usually 1-3 days. If it takes longer, feel free to ping the PR.

## Contact

- **GitHub Issues**: For bugs and features
- **Email**: josepemlengineer@gmail.com
- **Discussions**: For general questions and discussions

---

Thank you for contributing to NovaNN! 🚀 Your contribution helps this educational project continue to grow.
