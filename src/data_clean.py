import pandas as pd
from sklearn.impute import SimpleImputer
from utils.utils_data_clean import log_method, validate_columns_exist
from abstract_interfaces import AbstractDataFrameCleaner


class DataFrameCleaner(AbstractDataFrameCleaner):
    """
    Provides common data cleaning methods for a pandas DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input raw DataFrame to be cleaned.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @log_method
    def drop_missing(self, axis=0, how='any', thresh=None, subset=None):
        """
        Drop rows or columns with missing values.

        Parameters
        ----------
        axis : int
            0 to drop rows, 1 to drop columns.
        how : str
            'any' to drop if any missing, 'all' if all missing.
        thresh : int
            Require that many non-NA values.
        subset : list
            Specify columns to check for NA.

        Returns
        -------
        self
        """
        self.df.dropna(axis=axis, how=how, thresh=thresh, subset=subset, inplace=True)
        return self

    @log_method
    @validate_columns_exist
    def fill_missing(self, strategy='mean', columns=None):
        """
        Fill missing values using a specified strategy.

        Parameters
        ----------
        strategy : str
            One of 'mean', 'median', 'most_frequent', or 'constant'.
        columns : list
            Columns to apply imputation on.

        Returns
        -------
        self
        """
        if columns is None:
            columns = self.df.select_dtypes(include='number').columns
        imputer = SimpleImputer(strategy=strategy)
        self.df[columns] = imputer.fit_transform(self.df[columns])
        return self

    @log_method
    def drop_duplicates(self, subset=None, keep='first'):
        """
        Drop duplicate rows.

        Parameters
        ----------
        subset : list, optional
            Columns to consider for identifying duplicates.
        keep : str
            'first', 'last', or False to drop all duplicates.

        Returns
        -------
        self
        """
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return self

    @log_method
    def rename_columns(self, rename_dict):
        """
        Rename columns in the DataFrame.

        Parameters
        ----------
        rename_dict : dict
            Mapping from old column names to new ones.

        Returns
        -------
        self
        """
        self.df.rename(columns=rename_dict, inplace=True)
        return self

    @log_method
    @validate_columns_exist
    def drop_columns(self, columns):
        """
        Drop specified columns.

        Parameters
        ----------
        columns : list
            Column names to drop.

        Returns
        -------
        self
        """
        self.df.drop(columns=columns, inplace=True)
        return self

    @log_method
    def reset_index(self, drop=True):
        """
        Reset the index of the DataFrame.

        Parameters
        ----------
        drop : bool
            Drop the old index or not.

        Returns
        -------
        self
        """
        self.df.reset_index(drop=drop, inplace=True)
        return self

    @log_method
    def get_df(self):
        """
        Get the cleaned DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return self.df

    @log_method
    def to_csv(self, path, index=False):
        """
        Export the cleaned DataFrame to a CSV file.

        Parameters
        ----------
        path : str
            Destination file path.
        index : bool
            Whether to write row names (index).

        Returns
        -------
        self
        """
        self.df.to_csv(path, index=index)
        return self