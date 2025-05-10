import matplotlib.pyplot as plt
import streamlit as st
import io
from contextlib import contextmanager

@contextmanager
def streamlit_matplotlib_interceptor():
    """
    Intercepts plt.show() calls and reroutes the figure to Streamlit.
    """
    original_show = plt.show

    def fake_show(*args, **kwargs):
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    plt.show = fake_show
    try:
        yield
    finally:
        plt.show = original_show
