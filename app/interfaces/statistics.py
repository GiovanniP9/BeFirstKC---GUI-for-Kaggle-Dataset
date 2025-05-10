import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from app.utils.utils_statistics import log_method, validate_column_exists
from app.abstract.abstract_interfaces import AbstractStatisticalAnalyser


class StatisticalAnalyser(AbstractStatisticalAnalyser):
    """
    Perform basic and advanced statistical analyses on a pandas DataFrame.
    
    Includes methods for descriptive statistics, missing value analysis, 
    correlation analysis, distribution visualization, and outlier detection.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the analyzer with a DataFrame.
        
        Parameters:
        - dataframe (pd.DataFrame): The dataset to analyze.
        """
        self.df = dataframe

    @log_method
    def describe_data(self):
        """
        Return summary statistics for all numerical columns.

        Returns:
        - pd.DataFrame: Summary statistics including count, mean, std, min, max, quartiles.
        """
        return self.df.describe()

    @log_method
    def missing_values_summary(self):
        """
        Return the number and percentage of missing values for each column.

        Returns:
        - pd.DataFrame: Table with columns 'Missing Count' and 'Missing Percentage'.
        """
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100
        return pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage': missing_percent
        })

    @log_method
    def correlation_matrix(self, method='pearson', plot=False):
        """
        Compute the correlation matrix and optionally display a heatmap.

        Parameters:
        - method (str): Correlation method ('pearson', 'spearman', or 'kendall').
        - plot (bool): If True, displays a seaborn heatmap of the matrix.

        Returns:
        - pd.DataFrame: Correlation matrix.
        """
        corr = self.df.corr(method=method)
        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f"{method.capitalize()} Correlation Matrix")
            plt.show()
        return corr

    @log_method
    @validate_column_exists
    def distribution_plot(self, column: str):
        """
        Plot the distribution of a specified numerical column using a histogram with KDE.

        Parameters:
        - column (str): Column name to plot.
        """
        sns.histplot(self.df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    @log_method
    def skewness_kurtosis(self):
        """
        Calculate skewness and kurtosis for all numerical columns.

        Returns:
        - pd.DataFrame: Skewness and kurtosis values for each column.
        """
        return pd.DataFrame({
            'Skewness': self.df.skew(numeric_only=True),
            'Kurtosis': self.df.kurtosis(numeric_only=True)
        })

    @log_method
    @validate_column_exists
    def outlier_summary(self, column: str, method='iqr'):
        """
        Identify outliers in a column using IQR or Z-score methods.

        Parameters:
        - column (str): Column name to analyze.
        - method (str): Method to use: 'iqr' or 'zscore'.

        Returns:
        - pd.Series: Outlier values in the specified column.
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
        Generate count and percentage summaries for all categorical columns.

        Returns:
        - dict[str, pd.DataFrame]: Dictionary with column names as keys and
          DataFrames as values showing 'Count' and 'Percentage'.
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
