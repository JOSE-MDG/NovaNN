import os
from typing import Dict, Optional
from src._typing import InitFnArg

from dotenv import load_dotenv
from src.core.init import (
    calculate_gain,
    kaiming_normal_,
    kaiming_uniform_,
    random_init_,
    xavier_normal_,
    xavier_uniform_,
)

load_dotenv()


"""
Core configuration constants and default initialization maps.

This module exposes environment-configured paths and logger settings, plus
two dictionaries that map activation names to default weight initialization
functions.

Docstrings and type hints follow the Google style where applicable.
"""

FASHION_TRAIN_DATA_PATH: Optional[str] = os.getenv("FASHION_TRAIN_DATA_PATH")
EXPORTATION_FASHION_TRAIN_DATA_PATH: Optional[str] = os.getenv(
    "EXPORTATION_FASHION_TRAIN_DATA_PATH"
)
FASHION_TEST_DATA_PATH: Optional[str] = os.getenv("FASHION_TEST_DATA_PATH")
FASHION_VALIDATION_DATA_PATH: Optional[str] = os.getenv("FASHION_VALIDATION_DATA_PATH")

MNIST_TRAIN_DATA_PATH: Optional[str] = os.getenv("MNIST_TRAIN_DATA_PATH")
EXPORTATION_MNIST_TRAIN_DATA_PATH: Optional[str] = os.getenv(
    "EXPORTATION_MNIST_TRAIN_DATA_PATH"
)
MNIST_VALIDATION_DATA_PATH: Optional[str] = os.getenv("MNIST_VALIDATION_DATA_PATH")
MNIST_TEST_DATA_PATH: Optional[str] = os.getenv("MNIST_TEST_DATA_PATH")

# Logger config
LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
LOGGER_DEFAULT_FORMAT: Optional[str] = os.getenv("LOGGER_DEFAULT_FORMAT")
LOGGER_DEFAULT_LEVEL: Optional[str] = os.getenv("LOGGER_DEFAULT_LEVEL")
LOGGER_DATE_FORMAT: Optional[str] = os.getenv("LOGGER_DATE_FORMAT")

# Default initialization maps
# Keys correspond to activation identifiers used elsewhere in the project.
# Each value is a callable that receives a shape and applies an initialization.
DEFAULT_NORMAL_INIT_MAP: Dict[str, InitFnArg] = {
    "relu": lambda shape: kaiming_normal_(shape, a=0.0, nonlinearity="relu"),
    "leakyrelu": lambda shape: kaiming_normal_(shape, a=0.01, nonlinearity="leakyrelu"),
    "tanh": lambda shape: xavier_normal_(shape, gain=calculate_gain("tanh")),
    "sigmoid": lambda shape: xavier_normal_(shape, gain=calculate_gain("sigmoid")),
    "default": lambda shape: random_init_(shape),
}

DEFAULT_UNIFORM_INIT_MAP: Dict[str, InitFnArg] = {
    "relu": lambda shape: kaiming_uniform_(shape, a=0.0, nonlinearity="relu"),
    "leakyrelu": lambda shape: kaiming_uniform_(
        shape, a=0.01, nonlinearity="leakyrelu"
    ),
    "tanh": lambda shape: xavier_uniform_(shape, gain=calculate_gain("tanh")),
    "sigmoid": lambda shape: xavier_uniform_(shape, gain=calculate_gain("sigmoid")),
    "default": lambda shape: random_init_(shape),
}
