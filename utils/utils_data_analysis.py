import logging
from functools import wraps


# Setup Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def log_method(func):
    """Log the start and end of a method call."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__}")
        return result
    return wrapper

def validate_column_exists(func):
    """Ensure the specified column exists in the dataframe."""
    @wraps(func)
    def wrapper(self, column, *args, **kwargs):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")
        return func(self, column, *args, **kwargs)
    return wrapper

