import streamlit as st
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from app.core.base_interface_loader import discover_classes

st.set_page_config(layout="wide")
st.title("Dashboard Analisi e Machine Learning Interattiva")

uploaded_file = st.sidebar.file_uploader("Carica un file CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Anteprima del dataset")
    st.dataframe(df.head())

    # Caricamento dinamico delle classi
    classes = discover_classes()
    class_name = st.sidebar.selectbox("Seleziona una classe", list(classes.keys()))

    if class_name:
        klass = classes[class_name]
        obj = klass(df)

        # Docstring della classe
        doc_class = inspect.getdoc(klass)
        if doc_class:
            with st.expander("Descrizione della Classe"):
                st.markdown(f"> {doc_class}")

        st.subheader(f"Metodi disponibili in `{class_name}`")

        # Estrai metodi pubblici
        methods = {
            name: method for name, method in inspect.getmembers(obj, predicate=inspect.ismethod)
            if not name.startswith('_')
        }

        method_name = st.selectbox("Scegli un metodo", list(methods.keys()))

        if method_name:
            method = methods[method_name]

            # Docstring del metodo
            doc_method = inspect.getdoc(method)
            if doc_method:
                with st.expander("Descrizione del Metodo"):
                    st.markdown(f"> {doc_method}")

            sig = inspect.signature(method)
            kwargs = {}

            st.markdown("Inserisci i parametri del metodo (se richiesti):")
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
                elif annotation == str:
                    kwargs[param.name] = st.text_input(param.name)
                elif annotation == bool:
                    kwargs[param.name] = st.checkbox(param.name)
                elif annotation == int:
                    kwargs[param.name] = st.number_input(param.name, step=1)
                elif annotation == float:
                    kwargs[param.name] = st.number_input(param.name)
                else:
                    kwargs[param.name] = st.text_input(param.name)

            if st.button(f"Esegui `{method_name}`"):
                try:
                    result = method(**kwargs)

                    # Visualizza il risultato in base al tipo
                    st.markdown("Risultato")
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                    elif isinstance(result, pd.Series):
                        st.dataframe(result.to_frame())
                    elif hasattr(result, "__iter__") and not isinstance(result, str):
                        st.write(list(result))
                    elif result is not None:
                        st.write(result)

                    # Mostra i grafici generati
                    st.markdown("Grafico generato (se presente)")
                    st.pyplot(plt.gcf())
                    plt.clf()

                    # Pulsante per salvare i risultati
                    if isinstance(result, pd.DataFrame):
                        st.download_button(
                            label="Salva i risultati come CSV",
                            data=result.to_csv(index=False),
                            file_name="risultati.csv",
                            mime="text/csv"
                        )
                    elif hasattr(result, "__iter__") and not isinstance(result, str):
                        st.download_button(
                            label="Salva i risultati come CSV",
                            data=pd.DataFrame(list(result)).to_csv(index=False),
                            file_name="risultati.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"Errore nell'esecuzione del metodo: {e}")
