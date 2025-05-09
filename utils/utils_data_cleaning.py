import logging
from functools import wraps


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
