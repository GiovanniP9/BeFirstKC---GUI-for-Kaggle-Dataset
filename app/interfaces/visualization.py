import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from app.abstract.abstract_interfaces import AbstractVisualization
from app.utils.log_config import log_method 
from app.utils.utils_visualization import ensure_numeric_columns


class Visualization(AbstractVisualization):
    """Class for visualizing data."""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the Visualization object with a pandas DataFrame.

        Parameters:
        df: The dataset to visualize, should be a pandas DataFrame.
        
        Raises:
        TypeError: If the input is not a pandas DataFrame.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        self.df = dataframe.copy()

    @log_method
    def split_numerical_categorical(self):
        """
        Split the DataFrame into numerical and categorical columns.

        Returns:
        A tuple containing:
        - df_num: DataFrame with numerical columns (int, float).
        - df_cat: DataFrame with categorical columns (object, category).
        """
        df_num = self.df.select_dtypes(include=['int64', 'float64'])
        df_cat = self.df.select_dtypes(include=['object', 'category'])
        return df_num, df_cat

    @log_method
    def histplot(self):
        """
        Plot histograms for all columns in the DataFrame. 
        If the column is numerical, the plot will include a KDE (Kernel Density Estimate). 
        For categorical columns, it will show a histogram of the category frequencies.
        The histograms are displayed in a grid layout.
        """
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
        """
        Plot boxplots for all numerical columns in the DataFrame.
        Boxplots are displayed in a grid layout.
        """
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
    @ensure_numeric_columns
    def correlation_matrix(self):
        """
        Plot the correlation matrix for all numerical columns.
        The matrix will show pairwise correlations between numerical variables.
        A heatmap is used to visualize the correlations.
        """
        df_num, _ = self.split_numerical_categorical()
        matrix = df_num.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix")
        plt.show()

    @log_method
    def violin_plot(self):
        """
        Plot violin plots for all numerical columns.
        Violin plots combine aspects of boxplots and density plots to show 
        the distribution of numerical variables.
        The plots are displayed in a grid layout.
        """
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
