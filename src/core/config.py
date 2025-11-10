import os

from dotenv import load_dotenv
from src.core.init import (
    xavier_normal_,
    kaiming_normal_,
    xavier_uniform_,
    kaiming_uniform_,
    random_init_,
    calculate_gain,
)

load_dotenv()

# Data variables
FASHION_TRAIN_DATA_PATH = os.getenv("FASHION_TRAIN_DATA_PATH")
EXPORTATION_FASHION_TRAIN_DATA_PATH = os.getenv("EXPORTATION_FASHION_TRAIN_DATA_PATH")
FASHION_TEST_DATA_PATH = os.getenv("FASHION_TEST_DATA_PATH")
FASHION_VALIDATION_DATA_PATH = os.getenv("FASHION_VALIDATION_DATA_PATH")

MNIST_TRAIN_DATA_PATH = os.getenv("MNIST_TRAIN_DATA_PATH")
EXPORTATION_MNIST_TRAIN_DATA_PATH = os.getenv("EXPORTATION_MNIST_TRAIN_DATA_PATH")
MNIST_VALIDATION_DATA_PATH = os.getenv("MNIST_VALIDATION_DATA_PATH")
MNIST_TEST_DATA_PATH = os.getenv("MNIST_TEST_DATA_PATH")

# Logger config
LOG_FILE = os.getenv("LOG_FILE")
LOGGER_DEFAULT_FORMAT = os.getenv("LOGGER_DEFAULT_FORMAT")
LOGGER_DATE_FORMAT = os.getenv("LOGGER_DEFAULT_LEVEL")

# initializations dicts
DEFAULT_NORMAL_INIT_MAP = {
    "relu": lambda shape: kaiming_normal_(shape, a=0.0, nonlinearity="relu"),
    "leakyrelu": lambda shape: kaiming_normal_(shape, a=0.01, nonlinearity="leakyrelu"),
    "tanh": lambda shape: xavier_normal_(shape, gain=calculate_gain("tanh")),
    "sigmoid": lambda shape: xavier_normal_(shape, gain=calculate_gain("sigmoid")),
    "default": lambda shape: random_init_(shape),
}

DEFAULT_UNIFORM_INIT_MAP = {
    "relu": lambda shape: kaiming_uniform_(shape, a=0.0, nonlinearity="relu"),
    "leakyrelu": lambda shape: kaiming_uniform_(
        shape, a=0.01, nonlinearity="leakyrelu"
    ),
    "tanh": lambda shape: xavier_uniform_(shape, calculate_gain(nonlinearity="tanh")),
    "sigmoid": lambda shape: xavier_uniform_(
        shape, calculate_gain(nonlinearity="sigmoid")
    ),
    "default": lambda shape: random_init_(shape),
}

# Fashion mnist
FASHION_MANIST_LABELS_MAP = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
