import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.abstract_interfaces import AbstractVisualization
from utils.log_config import log_method 


class Visualization(AbstractVisualization):
    """ Classe per la visualizzazione dei dati."""
    def __init__(self, df: pd.DataFrame):
        # Inizializza la classe con un DataFrame
        self.df = df.copy()
    
    @log_method
    def split_numerical_categorical(self):
        # Divide il DataFrame in due: uno con variabili numeriche e uno con variabili categoriche
        df_num = self.df.select_dtypes(include=['int64', 'float64'])
        df_cat = self.df.select_dtypes(include=['object', 'category'])
        return df_num, df_cat
    
    @log_method
    def histplot(self, df: pd.DataFrame):
        # Crea istogrammi per tutte le colonne del DataFrame passato
        all_cols = df.columns
        n_cols = 3  # Numero di colonne nel layout dei subplot
        n_rows = (len(all_cols) + n_cols - 1) // n_cols  # Calcola il numero necessario di righe

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten()  # Appiattisce la griglia degli assi in un array 1D

        for i, col in enumerate(all_cols):
            # Usa histplot con KDE per variabili numeriche, altrimenti tratta come categoriche
            if df[col].dtype in ['int64', 'float64']:
                sns.histplot(df[col], ax=axes[i], kde=True)
            else:
                sns.histplot(df[col].astype(str), ax=axes[i], kde=False)
            axes[i].set_title(f'Histogram of {col}')  # Titolo del grafico

        # Rimuove assi inutilizzati
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Spaziatura tra i grafici
        plt.show()
    
    @log_method
    def boxplot(self, df: pd.DataFrame):
        # Crea boxplot per tutte le variabili numeriche
        df_num, df_cat = self.split_numerical_categorical()

        n_cols = 3
        n_rows = (len(df_num.columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(df_num.columns):
            sns.boxplot(x=df_num[col], ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')  # Titolo del grafico

        # Rimuove assi inutilizzati
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
    
    @log_method
    def correlation_matrix(self, df: pd.DataFrame):
        # Crea una matrice di correlazione tra variabili numeriche
        df_num, df_cat = self.split_numerical_categorical()

        matrix = df_num.corr()  # Calcola la matrice di correlazione
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)  # Heatmap con annotazioni
        plt.title("Correlation Matrix")
        plt.show()
    
    @log_method
    def violin_plot(self, df: pd.DataFrame):
        # Crea violin plot per tutte le variabili numeriche
        df_num, df_cat = self.split_numerical_categorical()

        n_cols = 3
        n_rows = (len(df_num.columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(df_num.columns):
            sns.violinplot(y=df_num[col], ax=axes[i])
            axes[i].set_title(f'Violin Plot of {col}')  # Titolo del grafico

        # Rimuove assi inutilizzati
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()