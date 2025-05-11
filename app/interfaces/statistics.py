import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from app.utils.utils_statistics import log_method, validate_column_exists
from app.abstract.abstract_interfaces import AbstractStatisticalAnalyser
from app.utils.utils_visualization import ensure_numeric_columns


class StatisticalAnalyser(AbstractStatisticalAnalyser):
    """
    Performs statistical analysis on a pandas DataFrame.

    This class includes methods for descriptive statistics, missing value analysis,
    correlation assessment, visual exploration of distributions, skewness and kurtosis
    computation, outlier detection, and categorical variable summaries.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the StatisticalAnalyser with a pandas DataFrame.

        Parameters:
        dataframe: The DataFrame containing the dataset to be analyzed.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        self.df = dataframe.copy()

    @log_method
    def describe_data(self):
        """
        Generate basic descriptive statistics for numerical columns.

        Returns:
        A DataFrame with count, mean, standard deviation, min, max, and quartiles.
        """
        return self.df.describe()

    @log_method
    def missing_values_summary(self):
        """
        Compute the total and percentage of missing values for each column.

        Returns:
        A DataFrame containing 'Missing Count' and 'Missing Percentage' for each column.
        """
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100
        return pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage': missing_percent
        })

    @log_method
    @ensure_numeric_columns
    def correlation_matrix(self, method='pearson', plot=False):
        """
        Compute the correlation matrix for numerical columns.

        Parameters:
        method: Correlation method to use ('pearson', 'spearman', or 'kendall').
        plot: If True, a heatmap of the correlation matrix is displayed.

        Returns:
        A DataFrame representing the correlation matrix.
        """
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")

        corr = self.df.corr(method=method)
        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f"{method.capitalize()} Correlation Matrix")
            plt.show()
        return corr

    @log_method
    @validate_column_exists
    def distribution_plot(self, column, save_path=None):
        """
        Plot the distribution of a numeric column with a histogram and KDE.

        Parameters:
        column: The name of the column to visualize.
        save_path: Optional path to save the plot to a file.
        """
        sns.histplot(self.df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @log_method
    def skewness_kurtosis(self):
        """
        Calculate skewness and kurtosis for numerical columns.

        Returns:
        A DataFrame containing skewness and kurtosis values for each numeric column.
        """
        return pd.DataFrame({
            'Skewness': self.df.skew(numeric_only=True),
            'Kurtosis': self.df.kurtosis(numeric_only=True)
        })

    @log_method
    @validate_column_exists
    def outlier_summary(self, column, method='iqr'):
        """
        Identify outliers in a specified numeric column using the IQR or Z-score method.

        Parameters:
        column: The name of the column to analyze.
        method: Method used for detection ('iqr' or 'zscore').

        Returns:
        A Series containing the detected outlier values.
        """
        data = self.df[column].dropna()

        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = data[(data < lower) | (data > upper)]
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
        else:
            raise ValueError("Invalid method: use 'iqr' or 'zscore'.")

        return outliers

    @log_method
    def categorical_summary(self):
        """
        Generate frequency and percentage tables for all categorical columns.

        Returns:
        A dictionary mapping each categorical column name to a DataFrame
        with 'Count' and 'Percentage' of each category.
        """
        summaries = {}
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            counts = self.df[col].value_counts(dropna=False)
            percentages = counts / len(self.df) * 100
            summaries[col] = pd.DataFrame({
                'Count': counts,
                'Percentage': percentages
            })
        return summaries
