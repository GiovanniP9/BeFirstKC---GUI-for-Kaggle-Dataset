import logging
from functools import wraps


def require_data(func):
    """Ensure that X and y are set before proceeding."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Use load_data() or set_features_and_target().")
        return func(self, *args, **kwargs)
    return wrapper


def require_split(func):
    """Ensure that data has been split before model training/evaluation."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data not split. Use split_data() first.")
        return func(self, *args, **kwargs)
    return wrapper


def require_model(func):
    """Ensure that a model has been set before training/evaluation."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.model is None:
            raise ValueError("Model not set. Use set_model() first.")
        return func(self, *args, **kwargs)
    return wrapper