"""Core configuration, initialization, and constant definitions for NovaNN.

This module provides essential utilities for weight initialization, configuration
management, and access to environment variables and dataset paths. It serves as
the foundation for consistent initialization and logging across the framework.
"""

# Import constants from constants.py
from .constants import (
    LOG_FILE,
    LOGGER_DATE_FORMAT,
    LOGGER_DEFAULT_FORMAT,
    LOGGER_DEFAULT_LEVEL,
    EXPORTATION_FASHION_TRAIN_DATA_PATH,
    EXPORTATION_MNIST_TRAIN_DATA_PATH,
    MNIST_TEST_DATA_PATH,
    MNIST_TRAIN_DATA_PATH,
    FASHION_TEST_DATA_PATH,
    FASHION_TRAIN_DATA_PATH,
    MNIST_VALIDATION_DATA_PATH,
    FASHION_VALIDATION_DATA_PATH,
    YAML_FILE_PATH,
)

__all__ = [
    "LOG_FILE",
    "LOGGER_DATE_FORMAT",
    "LOGGER_DEFAULT_FORMAT",
    "LOGGER_DEFAULT_LEVEL",
    "EXPORTATION_FASHION_TRAIN_DATA_PATH",
    "EXPORTATION_MNIST_TRAIN_DATA_PATH",
    "MNIST_TEST_DATA_PATH",
    "MNIST_TRAIN_DATA_PATH",
    "FASHION_TEST_DATA_PATH",
    "FASHION_TRAIN_DATA_PATH",
    "MNIST_VALIDATION_DATA_PATH",
    "FASHION_VALIDATION_DATA_PATH",
    "YAML_FILE_PATH",
]
