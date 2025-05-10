import inspect
import pandas as pd
from app.core.registry import CLASS_REGISTRY

def get_class_instance(class_name: str, df: pd.DataFrame):
    cls = CLASS_REGISTRY.get(class_name)
    if not cls:
        raise ValueError(f"Classe {class_name} non trovata nel registry.")
    return cls(df)

def get_methods(class_instance):
    methods = {}
    for name, func in inspect.getmembers(class_instance, predicate=inspect.ismethod):
        if not name.startswith("_"):
            sig = inspect.signature(func)
            methods[name] = sig
    return methods
