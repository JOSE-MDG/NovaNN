import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

"""Core configuration constants and default initialization maps.

Exposes environment-configured paths and logger settings  
"""

# Dataset paths
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

# Logger configuration
LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
LOGGER_DEFAULT_FORMAT: Optional[str] = os.getenv("LOGGER_DEFAULT_FORMAT")
LOGGER_DEFAULT_LEVEL: Optional[str] = os.getenv("LOGGER_DEFAULT_LEVEL")
LOGGER_DATE_FORMAT: Optional[str] = os.getenv("LOGGER_DATE_FORMAT")
