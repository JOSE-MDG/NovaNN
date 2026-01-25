"""Dataset loading functions for common machine learning datasets."""

from . import fashion, mnist

from .fashion import load_fashion_mnist_data
from .mnist import load_mnist_data

__all__ = ["load_fashion_mnist_data", "load_mnist_data", "fashion", "mnist"]
