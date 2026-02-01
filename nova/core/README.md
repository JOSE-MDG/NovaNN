# `core` Module

The **`core/`** module contains the central configuration and constants of the NovaNN framework. It manages dataset paths, download URLs, and logging configuration.

## Structure

```
core/
├── __init__.py      # Exports all constants and configuration
└── constants.py     # Defines constants using pathlib
```

## Files

### `constants.py`

Defines all framework-wide constants using `pathlib.Path`. All dataset paths resolve to `~/.novann/datasets/`, ensuring a consistent storage location across environments.

**Project Structure:**

- `PROJECT_ROOT`: Absolute path to the NovaNN project root, resolved relative to the location of `constants.py`.
- `DATA_ROOT`: Base directory for all datasets (`~/.novann/datasets/`).

**Dataset Paths — MNIST:**

- `MNIST_DIR`: `~/.novann/datasets/Mnist/`
- `MNIST_TRAIN_DATA_PATH`: Path to the full training set (`mnist_train.parquet`)
- `EXPORTATION_MNIST_TRAIN_DATA_PATH`: Path to the training set after validation split (`mnist_train_e.parquet`)
- `MNIST_TEST_DATA_PATH`: Path to the test set (`mnist_test.parquet`)
- `MNIST_VALIDATION_DATA_PATH`: Path to the validation set (`mnist_validation.parquet`)

**Dataset Paths — Fashion-MNIST:**

- `FASHION_DIR`: `~/.novann/datasets/FashionMnist/`
- `FASHION_TRAIN_DATA_PATH`: Path to the full training set (`fashion-mnist_train.parquet`)
- `EXPORTATION_FASHION_TRAIN_DATA_PATH`: Path to the training set after validation split (`fashion-mnist_train_e.parquet`)
- `FASHION_TEST_DATA_PATH`: Path to the test set (`fashion-mnist_test.parquet`)
- `FASHION_VALIDATION_DATA_PATH`: Path to the validation set (`fashion-mnist_validation.parquet`)

**Dataset URLs:**

- `MNIST_URLS`: Download URLs for the official MNIST IDX files (train/test images and labels).
- `FASHION_URLS`: Download URLs for the official Fashion-MNIST IDX files (train/test images and labels).

**Logger Configuration:**

- `LOG_FILE`: Path to the log file.. Default `None`
- `LOGGER_DEFAULT_FORMAT`: Log message format string
- `LOGGER_DATE_FORMAT`: Date format used in log entries

**Other:**

- `YAML_FILE_PATH`: Path to `native_functions.yaml`, used by the binding system.

### `__init__.py`

Exports all constants defined in `constants.py`, providing a clean import API from anywhere in the framework.

## Usage

### Accessing dataset paths

```python
from nova.core import MNIST_TRAIN_DATA_PATH, MNIST_TEST_DATA_PATH

print(MNIST_TRAIN_DATA_PATH)
# ~/.novann/datasets/Mnist/mnist_train.parquet

print(MNIST_TEST_DATA_PATH)
# ~/.novann/datasets/Mnist/mnist_test.parquet
```

### Accessing logging configuration

```python
from nova.core import LOGGER_DEFAULT_FORMAT, LOGGER_DATE_FORMAT
import logging

logging.basicConfig(
    filename="/path/to/logger_file.log",
    format=LOGGER_DEFAULT_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)
```

### Integration with other modules

The `core` module is used internally by:

- **`utils.logger`**: Reads logging configuration constants.
- **`utils.datasets.mnist`**: Reads MNIST path constants to load or trigger downloads.
- **`utils.datasets.fashion_mnist`**: Reads Fashion-MNIST path constants in the same way.
- **`_internal._binding`**: Reads `YAML_FILE_PATH` to load `native_functions.yaml`.

## Design

- **Centralization**: All paths and configuration live in a single file.
- **Predictability**: Paths always resolve to `~/.novann/datasets/`, regardless of where the framework is installed.
- **No external dependencies**: Configuration is defined directly using `pathlib.Path`. No `.env` files or environment variables required.
- **Separation of concerns**: `constants.py` defines, `__init__.py` exports.

## Best practices

1. **Don't hardcode paths**: Always import from `nova.core`.
2. **Validate before use**: Check that paths exist before reading files.
3. **Export new constants**: Any constant added to `constants.py` must also be exported in `__init__.py`.

---

> The `core` module is the configuration foundation of NovaNN. To add new constants, define them in `constants.py` and export them in `__init__.py`.
