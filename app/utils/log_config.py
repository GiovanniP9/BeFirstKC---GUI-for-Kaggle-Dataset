import logging
from functools import wraps
import time

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(ch)

def log_method(func):
    """Log the execution of DataFrameCleaner methods."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Running {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

def timer(time_weight=2.0):
    """Decorator to log the execution time of a function execution."""
    def _wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            stop = time.time()
            weighted_time = (stop - start) * time_weight
            logging.info(f"{func.__name__} executed in {weighted_time:.2f} seconds (weighted)")
            return res
        return wrapper
    return _wrapper

def repeat(times=3):
    """Decorator to repeat the execution of a function."""
    def _wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                func(*args, **kwargs)
        return wrapper
    return _wrapper