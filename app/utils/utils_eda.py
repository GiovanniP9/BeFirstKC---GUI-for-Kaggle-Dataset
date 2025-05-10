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
    """Log the execution of EDA methods."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__}")
        return result
    return wrapper


def validate_column_exists(func):
    """Check if provided columns exist in the dataframe."""
    @wraps(func)
    def wrapper(self, columns=None, *args, **kwargs):
        if columns is not None:
            missing = [col for col in columns if col not in self.df.columns]
            if missing:
                raise ValueError(f"The following columns were not found in the DataFrame: {missing}")
        return func(self, columns, *args, **kwargs)
    return wrapper