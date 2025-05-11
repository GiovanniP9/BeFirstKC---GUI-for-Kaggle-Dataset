import streamlit as st
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from app.core.base_interface_loader import discover_classes

st.set_page_config(layout="wide")
st.title("Dashboard Analisi e Machine Learning Interattiva")

uploaded_files = st.sidebar.file_uploader("Carica uno o più file CSV", type=["csv"], accept_multiple_files=True)

dfs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        dfs.append(df)
        st.write(f"Anteprima del dataset {uploaded_file.name}")
        st.dataframe(df.head())
    combined_df = pd.concat(dfs, ignore_index=True)
else:
    combined_df = None

# Caricamento dinamico delle classi
classes = discover_classes()
class_name = st.sidebar.selectbox("Seleziona una classe", list(classes.keys()))

if class_name:
    klass = classes[class_name]
    constructor_sig = inspect.signature(klass.__init__)

    constructor_params = {}
    st.sidebar.markdown("### Parametri del costruttore")
    for name, param in constructor_sig.parameters.items():
        if name == "self":
            continue
        default = param.default
        annotation = param.annotation

        key = f"constructor_{name}"

        if isinstance(default, bool):
            constructor_params[name] = st.sidebar.checkbox(name, value=default)
        elif isinstance(default, int):
            constructor_params[name] = st.sidebar.number_input(name, value=default)
        elif isinstance(default, float):
            constructor_params[name] = st.sidebar.number_input(name, value=default)
        elif isinstance(default, str):
            constructor_params[name] = st.sidebar.text_input(name, value=default)
        elif annotation == pd.DataFrame and combined_df is not None:
            constructor_params[name] = combined_df
        else:
            constructor_params[name] = st.sidebar.text_input(name)

    try:
        obj = klass(**constructor_params)
    except Exception as e:
        st.error(f"Errore nella creazione dell'oggetto: {e}")
        obj = None

    if obj:
        # Docstring della classe
        doc_class = inspect.getdoc(klass)
        if doc_class:
            with st.expander("Descrizione della Classe"):
                st.markdown(f"> {doc_class}")

        st.subheader(f"Metodi disponibili in `{class_name}`")

        methods = {
            name: method for name, method in inspect.getmembers(obj, predicate=inspect.ismethod)
            if not name.startswith('_')
        }

        method_name = st.selectbox("Scegli un metodo", list(methods.keys()))
        if method_name:
            method = methods[method_name]

            doc_method = inspect.getdoc(method)
            if doc_method:
                with st.expander("Descrizione del Metodo"):
                    st.markdown(f"> {doc_method}")

            sig = inspect.signature(method)
            kwargs = {}

            st.markdown("Inserisci i parametri del metodo:")
            for param in sig.parameters.values():
                if param.name == "self":
                    continue
                default = param.default
                annotation = param.annotation

                if isinstance(default, bool):
                    kwargs[param.name] = st.checkbox(param.name, value=default)
                elif isinstance(default, int):
                    kwargs[param.name] = st.number_input(param.name, value=default)
                elif isinstance(default, float):
                    kwargs[param.name] = st.number_input(param.name, value=default)
                elif isinstance(default, str):
                    kwargs[param.name] = st.text_input(param.name, value=default)
                elif annotation == pd.DataFrame and combined_df is not None:
                    kwargs[param.name] = combined_df
                else:
                    kwargs[param.name] = st.text_input(param.name)

            if st.button(f"Esegui `{method_name}`"):
                try:
                    result = method(**kwargs)

                    st.markdown("### Risultato")
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                    elif isinstance(result, pd.Series):
                        st.dataframe(result.to_frame())
                    elif hasattr(result, "__iter__") and not isinstance(result, str):
                        st.write(list(result))
                    elif result is not None:
                        st.write(result)

                    # Mostra eventuali grafici
                    st.markdown("### Grafico generato (se presente)")
                    st.pyplot(plt.gcf())
                    plt.clf()

                    # Salva i risultati
                    if isinstance(result, pd.DataFrame):
                        st.download_button(
                            label="Salva i risultati come CSV",
                            data=result.to_csv(index=False),
                            file_name="risultati.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Errore nell'esecuzione del metodo: {e}")
