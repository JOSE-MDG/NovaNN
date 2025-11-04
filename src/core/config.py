import os

from dotenv import load_dotenv

load_dotenv()

# Data variables
FASHION_TRAIN_DATA_PATH = os.getenv("FASHION_TRAIN_DATA_PATH")
FASHION_TEST_DATA_PATH = os.getenv("FASHION_TEST_DATA_PATH")
MNIST_TRAIN_DATA_PATH = os.getenv("MNIST_TRAIN_DATA_PATH")
MNIST_TEST_DATA_PATH = os.getenv("MNIST_TEST_DATA_PATH")
