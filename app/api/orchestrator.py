import inspect
import pandas as pd
from app.core.registry import CLASS_REGISTRY
from utils.plot_interceptor import streamlit_matplotlib_interceptor
import streamlit as st
import inspect


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


def invoke_method(instance, method_name, params):
    """
    Esegue un metodo su una classe data, con i parametri da interfaccia.
    Intercetta eventuali grafici per Streamlit.
    """
    try:
        method = getattr(instance, method_name)
        signature = inspect.signature(method)

        # Valida i parametri richiesti
        final_args = {}
        for param_name, param in signature.parameters.items():
            if param_name in params:
                final_args[param_name] = params[param_name]
            elif param.default is inspect.Parameter.empty:
                raise ValueError(f"Parametro richiesto: {param_name}")

        # Intercetta grafici se esistono
        with streamlit_matplotlib_interceptor():
            result = method(**final_args)

        # Mostra risultati se non sono grafici
        if result is not None:
            st.write(result)

    except Exception as e:
        st.error(f"Errore durante l'esecuzione del metodo `{method_name}`: {e}")
