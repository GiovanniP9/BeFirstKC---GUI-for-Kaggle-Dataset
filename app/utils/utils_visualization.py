from functools import wraps
import pandas as pd

def ensure_numeric_columns(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        df = self.df
        numeric_df = df.select_dtypes(include=['number'])
        self.df = numeric_df  # sostituzione temporanea
        result = method(self, *args, **kwargs)
        self.df = df  # ripristina il DataFrame originale
        return result
    return wrapper
