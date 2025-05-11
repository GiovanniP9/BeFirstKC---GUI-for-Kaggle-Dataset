import pandas as pd
from sklearn.impute import SimpleImputer
from app.utils.utils_cleaner import log_method, validate_columns_exist
from app.abstract.abstract_interfaces import AbstractDataFrameCleaner


class DataFrameCleaner(AbstractDataFrameCleaner):
    """
    A class for performing common data cleaning operations on a pandas DataFrame.
    
    Attributes
    ----------
    df : pd.DataFrame
        Internal copy of the input DataFrame used for transformations.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataFrameCleaner with a copy of the provided DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            The raw input DataFrame to clean.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        self.df = dataframe.copy()

    @log_method
    def drop_missing(self, axis=0, how='any', thresh=None, subset=None):
        """
        Drop rows or columns with missing (NaN) values.

        Parameters
        ----------
        axis : int, default=0
            0 to drop rows, 1 to drop columns.
        how : str, default='any'
            'any' drops if any value is missing, 'all' drops if all are missing.
        thresh : int, optional
            Require that many non-NA values to retain the row/column.
        subset : list of str, optional
            Labels along the axis to consider for NA checks.

        Returns
        -------
        self : DataFrameCleaner
            The updated cleaner instance.
        """
        self.df.dropna(axis=axis, how=how, thresh=thresh, subset=subset, inplace=True)
        return self

    @log_method
    @validate_columns_exist
    def fill_missing(self, strategy='mean', columns=None):
        """
        Fill missing values using a specified imputation strategy.

        Parameters
        ----------
        strategy : str, default='mean'
            Strategy to use: 'mean', 'median', 'most_frequent', or 'constant'.
        columns : list of str, optional
            Columns to apply the imputation. Defaults to all numeric columns.

        Returns
        -------
        self : DataFrameCleaner
            The updated cleaner instance.
        """
        if columns is None:
            columns = self.df.select_dtypes(include='number').columns
        
        # Impute missing values with the chosen strategy
        imputer = SimpleImputer(strategy=strategy)
        self.df[columns] = imputer.fit_transform(self.df[columns])
        return self

    @log_method
    def drop_duplicates(self, subset=None, keep='first'):
        """
        Remove duplicate rows from the DataFrame.

        Parameters
        ----------
        subset : list of str, optional
            Columns to consider for identifying duplicates.
        keep : {'first', 'last', False}, default='first'
            Determines which duplicates to keep.

        Returns
        -------
        self : DataFrameCleaner
            The updated cleaner instance.
        """
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return self

    @log_method
    def rename_columns(self, rename_dict):
        """
        Rename one or more columns using a mapping dictionary.

        Parameters
        ----------
        rename_dict : dict
            Dictionary mapping existing column names to new names.

        Returns
        -------
        self : DataFrameCleaner
            The updated cleaner instance.
        """
        self.df.rename(columns=rename_dict, inplace=True)
        return self

    @log_method
    @validate_columns_exist
    def drop_columns(self, columns):
        """
        Remove specified columns from the DataFrame.

        Parameters
        ----------
        columns : list of str
            Column names to be dropped.

        Returns
        -------
        self : DataFrameCleaner
            The updated cleaner instance.
        """
        self.df.drop(columns=columns, inplace=True)
        return self

    @log_method
    def reset_index(self, drop=True):
        """
        Reset the index of the DataFrame.

        Parameters
        ----------
        drop : bool, default=True
            If True, do not insert the old index as a column.

        Returns
        -------
        self : DataFrameCleaner
            The updated cleaner instance.
        """
        self.df.reset_index(drop=drop, inplace=True)
        return self

    @log_method
    def get_df(self):
        """
        Return the cleaned DataFrame.

        Returns
        -------
        pd.DataFrame
            The current state of the DataFrame.
        """
        return self.df

    @log_method
    def to_csv(self, path, index=False):
        """
        Save the cleaned DataFrame to a CSV file.

        Parameters
        ----------
        path : str
            File path to write the CSV to.
        index : bool, default=False
            Whether to include the index in the CSV output.

        Returns
        -------
        self : DataFrameCleaner
            The cleaner instance, unchanged.
        """
        self.df.to_csv(path, index=index)
        return self
