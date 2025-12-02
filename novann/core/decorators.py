import time
from functools import wraps
from novann.core import logger


def measure_exc_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        reusul = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"Execution time of '{func.__name__}' was {round(end - start)}s")
        del reusul

    return wrapper
