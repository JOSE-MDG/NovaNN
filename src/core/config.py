import os

from dotenv import load_dotenv
from src.core.init import (
    xavier_normal_,
    kaiming_normal_,
    xavier_uniform_,
    kaiming_uniform_,
    random_init_,
)

load_dotenv()

# Data variables
FASHION_TRAIN_DATA_PATH = os.getenv("FASHION_TRAIN_DATA_PATH")
FASHION_TEST_DATA_PATH = os.getenv("FASHION_TEST_DATA_PATH")
FASHION_VALIDATION_DATA_PATH = os.getenv("FASHION_VALIDATION_DATA_PATH")
MNIST_TRAIN_DATA_PATH = os.getenv("MNIST_TRAIN_DATA_PATH")
MNIST_VALIDATION_DATA_PATH = os.getenv("MNIST_VALIDATION_DATA_PATH")
MNIST_TEST_DATA_PATH = os.getenv("MNIST_TEST_DATA_PATH")

# Logger config
LOG_FILE = os.getenv("LOG_FILE")
LOGGER_DEFAULT_FORMAT = os.getenv("LOGGER_DEFAULT_FORMAT")
LOGGER_DATE_FORMAT = os.getenv("LOGGER_DEFAULT_LEVEL")

# initializations dicts
DEFAULT_NORMAL_INIT_MAP = {
    "relu": kaiming_normal_,
    "tanh": xavier_normal_,
    "sigmoid": xavier_normal_,
    "default": random_init_,
}

DEFAULT_UNIFORM_INIT_MAP = {
    "relu": kaiming_uniform_,
    "tanh": xavier_uniform_,
    "sigmoid": xavier_uniform_,
    "default": random_init_,
}
