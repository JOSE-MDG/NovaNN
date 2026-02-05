"""Core configuration constants

Simple, predictable paths for datasets and logging.
All datasets are stored in ~/.novann/datasets/
"""

from pathlib import Path


# Project Structure


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# Dataset Storage - All datasets go to ~/.novann/datasets/


DATA_ROOT = Path.home() / ".novann" / "datasets"

# MNIST paths
MNIST_DIR = DATA_ROOT / "Mnist"
MNIST_TRAIN_DATA_PATH = MNIST_DIR / "mnist_train.parquet"
EXPORTATION_MNIST_TRAIN_DATA_PATH = MNIST_DIR / "mnist_train_e.parquet"
MNIST_TEST_DATA_PATH = MNIST_DIR / "mnist_test.parquet"
MNIST_VALIDATION_DATA_PATH = MNIST_DIR / "mnist_validation.parquet"

# Fashion-MNIST paths
FASHION_DIR = DATA_ROOT / "FashionMnist"
FASHION_TRAIN_DATA_PATH = FASHION_DIR / "fashion-mnist_train.parquet"
EXPORTATION_FASHION_TRAIN_DATA_PATH = FASHION_DIR / "fashion-mnist_train_e.parquet"
FASHION_TEST_DATA_PATH = FASHION_DIR / "fashion-mnist_test.parquet"
FASHION_VALIDATION_DATA_PATH = FASHION_DIR / "fashion-mnist_validation.parquet"


# Dataset URLs


MNIST_URLS = {
    "train-images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test-images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test-labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

FASHION_URLS = {
    "train-images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train-labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test-images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test-labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}


# Logger Configuration
LOGGER_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s| %(name)s : %(funcName)s - %(message)s"
)
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# Other Configuration

YAML_FILE_PATH = (
    PROJECT_ROOT / "nova" / "autograd" / "_ops" / "native" / "native_functions.yaml"
)
