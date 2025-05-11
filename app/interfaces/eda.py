import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from app.utils.utils_eda import log_method, validate_column_exists
from app.abstract.abstract_interfaces import AbstractEDA


class EDA(AbstractEDA):
    """
    Perform basic Exploratory Data Analysis (EDA) on a pandas DataFrame.
    
    This class provides tools to:
    - inspect general dataset structure and missing data
    - summarize numerical and categorical columns
    - visualize feature distributions, outliers, and correlations
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the EDA object with a pandas DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input dataset to analyze.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        self.df = dataframe.copy()

    @log_method
    def check_basic_info(self) -> None:
        """
        Print basic information about the dataset, including:
        - shape
        - structure (dtypes, null counts)
        - missing values
        - descriptive statistics for all columns
        """
        print("Shape:", self.df.shape)
        print("\nInfo:")
        print(self.df.info())  # Prints structure of the DataFrame
        print("\nMissing values:")
        print(self.df.isnull().sum())
        print("\nDescriptive statistics:")
        print(self.df.describe(include='all'))

    @log_method
    def summarize_numerical(self) -> pd.DataFrame:
        """
        Return summary statistics for all numerical columns.

        Returns
        -------
        pd.DataFrame
            Descriptive stats (count, mean, std, etc.) for numerical features.
        """
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        return self.df[num_cols].describe()

    @log_method
    def summarize_categorical(self) -> pd.DataFrame:
        """
        Return summary statistics for all categorical columns.

        Returns
        -------
        pd.DataFrame
            Statistics including count, unique, top, and frequency for categorical features.
        """
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        return self.df[cat_cols].describe()

    @log_method
    @validate_column_exists
    def plot_histograms(self, columns=None) -> None:
        """
        Plot histograms for specified or all numerical columns.

        Parameters
        ----------
        columns : list of str, optional
            List of numerical columns to plot. If None, plots all numeric columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        self.df[columns].hist(bins=30, figsize=(15, 10))
        plt.tight_layout()
        plt.show()

    @log_method
    @validate_column_exists
    def plot_boxplots(self, columns=None) -> None:
        """
        Plot boxplots for specified or all numerical columns to detect outliers.

        Parameters
        ----------
        columns : list of str, optional
            Columns to plot. If None, plots all numeric columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns

        for col in columns:
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()

    @log_method
    def plot_correlation_matrix(self) -> None:
        """
        Plot a heatmap of the correlation matrix between numerical features.
        """
        num_df = self.df.select_dtypes(include=['int64', 'float64'])
        corr = num_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    @log_method
    def plot_target_relations(self, target: str) -> None:
        """
        Plot scatterplots showing relationships between each numerical feature and a target column.

        Parameters
        ----------
        target : str
            Name of the target column to compare against other features.

        Raises
        ------
        ValueError
            If the target column is not found in the DataFrame.
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
                plt.tight_layout()
                plt.show()
