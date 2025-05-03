import logging
from functools import wraps

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

def validate_columns_exist(func):
    """Decorator to validate that specified columns exist in the DataFrame."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        columns = kwargs.get('columns') or (args[0] if args else None)
        if columns is not None:
            missing = [col for col in columns if col not in self.df.columns]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
        return func(self, *args, **kwargs)
    return wrapper
