import streamlit as st
import inspect
import pandas as pd
from app.core.base_interface_loader import discover_classes

st.set_page_config(layout="wide")

st.title("🧠 Dashboard Analisi e ML Interattiva")

uploaded_file = st.sidebar.file_uploader("Carica un file CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Anteprima del dataset:", df.head())

    # Load classi dinamicamente
    classes = discover_classes()
    class_name = st.sidebar.selectbox("📦 Seleziona Classe", list(classes.keys()))

    if class_name:
        klass = classes[class_name]
        obj = klass(df)

        st.subheader(f"🛠️ Metodi disponibili in `{class_name}`")

        # Estrai tutti i metodi pubblici
        methods = {
            name: method for name, method in inspect.getmembers(obj, predicate=inspect.ismethod)
            if not name.startswith('_')
        }

        method_name = st.selectbox("🔧 Scegli un metodo", list(methods.keys()))

        if method_name:
            method = methods[method_name]
            sig = inspect.signature(method)

            kwargs = {}
            for param in sig.parameters.values():
                if param.name == "self":
                    continue
                default = param.default
                if isinstance(default, bool):
                    kwargs[param.name] = st.checkbox(param.name, value=default)
                elif isinstance(default, int):
                    kwargs[param.name] = st.number_input(param.name, value=default)
                elif isinstance(default, float):
                    kwargs[param.name] = st.number_input(param.name, value=default)
                elif isinstance(default, str):
                    kwargs[param.name] = st.text_input(param.name, value=default)
                elif param.annotation == str and param.default is inspect.Parameter.empty:
                    kwargs[param.name] = st.text_input(param.name)
                elif param.annotation == bool:
                    kwargs[param.name] = st.checkbox(param.name)
                elif param.annotation == int:
                    kwargs[param.name] = st.number_input(param.name, step=1)
                elif param.annotation == float:
                    kwargs[param.name] = st.number_input(param.name)
                else:
                    kwargs[param.name] = st.text_input(param.name)

            if st.button(f"Esegui `{method_name}`"):
                result = method(**kwargs)

                # Output in base al tipo
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif isinstance(result, pd.Series):
                    st.write(result.to_frame())
                elif hasattr(result, "__iter__") and not isinstance(result, str):
                    st.write(list(result))
                elif result is not None:
                    st.write(result)
