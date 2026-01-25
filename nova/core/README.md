# `core` Module

The **`core/`** module contains the central configuration and constants of the NovaNN framework. It manages environment variables, dataset paths, and logging system configuration.

## Structure

```
core/
├── __init__.py      # Exports all constants and configuration
└── constants.py     # Defines constants and loads environment variables
```

## Files

### `constants.py`

Loads environment variables using `dotenv` and exposes configurable constants for the framework.

**Environment Variables - Datasets:**

- **Fashion-MNIST:**
  - `FASHION_TRAIN_DATA_PATH`: Path to the training set
  - `EXPORTATION_FASHION_TRAIN_DATA_PATH`: Path for exporting processed data
  - `FASHION_TEST_DATA_PATH`: Path to the test set
  - `FASHION_VALIDATION_DATA_PATH`: Path to the validation set

- **MNIST:**
  - `MNIST_TRAIN_DATA_PATH`: Path to the training set
  - `EXPORTATION_MNIST_TRAIN_DATA_PATH`: Path for exporting processed data
  - `MNIST_TEST_DATA_PATH`: Path to the test set
  - `MNIST_VALIDATION_DATA_PATH`: Path to the validation set

**Environment Variables - Logging:**

- `LOG_FILE`: Path to the log file
- `LOGGER_DEFAULT_FORMAT`: Log message format
- `LOGGER_DEFAULT_LEVEL`: Default logging level (DEBUG, INFO, WARNING, ERROR)
- `LOGGER_DATE_FORMAT`: Date format in logs

**Environment Variables - System:**

- `NATIVE_YAML`: Path to the `native_functions.yaml` file for the binding system

### `__init__.py`

Exports all constants defined in `constants.py`, providing a clean API to access configuration from any part of the framework.

## Usage

### Configuration with `.env` file

Create a `.env` file in the project root:

```env
# Datasets
FASHION_TRAIN_DATA_PATH=/path/to/fashion_mnist/train
FASHION_TEST_DATA_PATH=/path/to/fashion_mnist/test
MNIST_TRAIN_DATA_PATH=/path/to/mnist/train
MNIST_TEST_DATA_PATH=/path/to/mnist/test

# Logging
LOG_FILE=logs/nova.log
LOGGER_DEFAULT_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOGGER_DEFAULT_LEVEL=INFO
LOGGER_DATE_FORMAT=%Y-%m-%d %H:%M:%S

# System
NATIVE_YAML=nova/autograd/_ops/native/native_functions.yaml
```

### Accessing constants

```python
from nova.core import (
    MNIST_TRAIN_DATA_PATH,
    LOG_FILE,
    LOGGER_DEFAULT_LEVEL
)

# Use dataset paths
train_path = MNIST_TRAIN_DATA_PATH

# Configure logging
import logging
logging.basicConfig(
    filename=LOG_FILE,
    level=LOGGER_DEFAULT_LEVEL,
    format=LOGGER_DEFAULT_FORMAT
)
```

### Integration with other modules

The `core` module is used internally by:

- **`utils.logger`**: Reads logging configuration
- **`_internal._binding`**: Reads `YAML_FILE_PATH` to load `native_functions.yaml`
- **Training scripts**: Access dataset paths

## Design

The `core` module follows these principles:

- **Centralization**: All constants and configuration in one place
- **Flexibility**: Configurable variables via `.env` without modifying code
- **Type hints**: All constants typed as `Optional[str]`
- **Separation of concerns**: `constants.py` defines, `__init__.py` exports
- **Safe defaults**: Variables can be `None` if not defined

## Optional variables

All constants are `Optional[str]`, which allows:

```python
from nova.core import MNIST_TRAIN_DATA_PATH

if MNIST_TRAIN_DATA_PATH is None:
    print("⚠️ MNIST path not configured. Using default.")
    MNIST_TRAIN_DATA_PATH = "./data/mnist/train"
```

## Best practices

1. **Don't hardcode paths**: Always use `core` constants
2. **Validate before use**: Verify that paths exist
3. **Document new variables**: Add comments in `constants.py`
4. **Keep `.env` local**: Never commit `.env` to the repository

---

> The `core` module is the configuration foundation of NovaNN. To add new constants, edit them in `constants.py` and export them in `__init__.py`.
