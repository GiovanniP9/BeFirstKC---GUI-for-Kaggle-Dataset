import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from app.abstract.abstract_interfaces import AbstractVisualization
from app.utils.log_config import log_method 


class Visualization(AbstractVisualization):
    """Classe per la visualizzazione dei dati."""
    
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        self.df = df

    @log_method
    def split_numerical_categorical(self):
        df_num = self.df.select_dtypes(include=['int64', 'float64'])
        df_cat = self.df.select_dtypes(include=['object', 'category'])
        return df_num, df_cat

    @log_method
    def histplot(self):
        df = self.df
        all_cols = df.columns
        n_cols = 3
        n_rows = (len(all_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(all_cols):
            if df[col].dtype in ['int64', 'float64']:
                sns.histplot(df[col], ax=axes[i], kde=True)
            else:
                sns.histplot(df[col].astype(str), ax=axes[i], kde=False)
            axes[i].set_title(f'Histogram of {col}')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        plt.show()

    @log_method
    def boxplot(self):
        df_num, _ = self.split_numerical_categorical()

        n_cols = 3
        n_rows = (len(df_num.columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(df_num.columns):
            sns.boxplot(x=df_num[col], ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    @log_method
    def correlation_matrix(self):
        df_num, _ = self.split_numerical_categorical()
        matrix = df_num.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix")
        plt.show()

    @log_method
    def violin_plot(self):
        df_num, _ = self.split_numerical_categorical()

        n_cols = 3
        n_rows = (len(df_num.columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(df_num.columns):
            sns.violinplot(y=df_num[col], ax=axes[i])
            axes[i].set_title(f'Violin Plot of {col}')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()