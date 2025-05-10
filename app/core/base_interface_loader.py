import inspect
import importlib
import pkgutil
from app.interfaces import __path__ as interfaces_path

def discover_classes():
    classes = {}
    for _, module_name, _ in pkgutil.iter_modules(interfaces_path):
        module = importlib.import_module(f"app.interfaces.{module_name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name not in classes and obj.__module__.startswith("app.interfaces"):
                classes[name] = obj
    return classes
