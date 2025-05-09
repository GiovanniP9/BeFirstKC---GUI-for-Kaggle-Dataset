import logging
from functools import wraps


def validate_column_exists(func):
    """Ensure the specified column exists in the dataframe."""
    @wraps(func)
    def wrapper(self, column, *args, **kwargs):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")
        return func(self, column, *args, **kwargs)
    return wrapper

