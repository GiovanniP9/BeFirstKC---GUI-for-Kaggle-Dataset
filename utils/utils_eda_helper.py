import logging
from functools import wraps


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