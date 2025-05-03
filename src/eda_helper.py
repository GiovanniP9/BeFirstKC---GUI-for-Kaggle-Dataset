import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils_eda_helper import log_method, validate_column_exists
from abstract_interfaces import AbstractEDA


class EDA(AbstractEDA):
    """
    Perform basic Exploratory Data Analysis (EDA) on a pandas DataFrame.
    Includes data inspection, summary statistics, and plotting utilities.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    @log_method
    def check_basic_info(self):
        """Print basic info about the dataset: shape, info, nulls, and descriptive statistics."""
        print("Shape:", self.df.shape)
        print("\nInfo:")
        print(self.df.info())
        print("\nMissing values:")
        print(self.df.isnull().sum())
        print("\nDescriptive statistics:")
        print(self.df.describe(include='all'))

    @log_method
    def summarize_numerical(self):
        """Return summary statistics for numerical columns."""
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        return self.df[num_cols].describe()

    @log_method
    def summarize_categorical(self):
        """Return summary statistics for categorical columns."""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        return self.df[cat_cols].describe()

    @log_method
    @validate_column_exists
    def plot_histograms(self, columns=None):
        """Plot histograms for the specified or all numerical columns."""
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.df[columns].hist(bins=30, figsize=(15, 10))
        plt.tight_layout()
        plt.show()

    @log_method
    @validate_column_exists
    def plot_boxplots(self, columns=None):
        """Plot boxplots for the specified or all numerical columns."""
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in columns:
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot of {col}')
            plt.show()

    @log_method
    def plot_correlation_matrix(self):
        """Plot a correlation matrix heatmap for numerical features."""
        num_df = self.df.select_dtypes(include=['int64', 'float64'])
        corr = num_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

    @log_method
    def plot_target_relations(self, target):
        """
        Plot scatterplots between numerical features and the target column.
        
        Parameters:
        target (str): Name of the target column.
        """
        if target not in self.df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe.")
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            if col != target:
                sns.scatterplot(x=self.df[col], y=self.df[target])
                plt.title(f'{col} vs {target}')
                plt.xlabel(col)
                plt.ylabel(target)
                plt.show()
