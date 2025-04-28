import pandas as pd
from sklearn.impute import SimpleImputer
from abstract_interfaces import AbstractDataFrameCleaner


class DataFrameCleaner(AbstractDataFrameCleaner):
    """Represents the data cleaner class that turns a dataframe
    into a cleaner dataframe using the following methods:
    
    Args:
        AbstractDataFrameCleaner (pd.Dataframe): cleaned dataframe
    """
    def __init__(self, df: pd.DataFrame):
        """Initialize a shallow copy of your dataframe

        Args:
            df (pd.DataFrame): not cleaned dataframe
        """
        self.df = df.copy()
    
    def drop_missing(self, axis=0, how='any', thresh=None, subset=None):
        """Drop missing values."""
        self.df.dropna(axis=axis, how=how, thresh=thresh, subset=subset, inplace=True)
        return self

    def fill_missing(self, strategy='mean', columns=None):
        """Fill missing values with a given strategy."""
        if columns is None:
            columns = self.df.select_dtypes(include='number').columns
        imputer = SimpleImputer(strategy=strategy)
        self.df[columns] = imputer.fit_transform(self.df[columns])
        return self

    def drop_duplicates(self, subset=None, keep='first'):
        """Drop duplicate rows."""
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return self

    def rename_columns(self, rename_dict):
        """Rename columns based on a dictionary."""
        self.df.rename(columns=rename_dict, inplace=True)
        return self

    def drop_columns(self, columns):
        """Drop specified columns."""
        self.df.drop(columns=columns, inplace=True)
        return self

    def get_df(self):
        """Return the cleaned dataframe."""
        return self.df

    def reset_index(self, drop=True):
        """Reset the index of the DataFrame."""
        self.df.reset_index(drop=drop, inplace=True)
        return self

    def to_csv(self, path, index=False):
        """Save the cleaned DataFrame to a CSV file."""
        self.df.to_csv(path, index=index)
        return self

