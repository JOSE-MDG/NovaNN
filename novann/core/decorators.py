import time
from functools import wraps
from novann.core import logger


def measure_exc_time(func):
    """Decorator function to measure the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wraps the original function"""

        # start time execution
        start = time.perf_counter()
        reusul = func(*args, **kwargs)
        end = time.perf_counter()

        logger.info(f"Execution time of '{func.__name__}' was {round(end - start)}s")
        del reusul  # remove the result of the function

    return wrapper
