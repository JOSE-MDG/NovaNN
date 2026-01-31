"""Dataset loading functions for common machine learning datasets."""

from . import fashion, mnist

from .fashion import load_fashion_mnist_data, load_fashion_mnist_default
from .mnist import load_mnist_data, load_mnist_default

__all__ = [
    "load_fashion_mnist_data",
    "load_fashion_mnist_default",
    "load_mnist_data",
    "load_mnist_default",
    "fashion",
    "mnist",
]
