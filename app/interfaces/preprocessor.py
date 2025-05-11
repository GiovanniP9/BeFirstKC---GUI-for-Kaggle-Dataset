import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from app.abstract.abstract_interfaces import AbstractPreprocessor


class Preprocessor(AbstractPreprocessor):
    """
    Handles common preprocessing tasks:
    - Missing value handling
    - Categorical encoding
    - Feature scaling
    - Train-test splitting
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with a copy of the provided DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The raw dataset to preprocess.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        self.df = dataframe.copy()
        self.scaler = None

    def fill_missing(self, strategy: str = 'mean') -> None:
        """
        Fill or drop missing values using the given strategy.

        Parameters
        ----------
        strategy : {'mean', 'median', 'mode', 'drop'}
            Strategy to fill missing values.

        Raises
        ------
        ValueError
            If an unsupported strategy is passed.
        """
        if strategy == 'mean':
            self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
        elif strategy == 'median':
            self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        elif strategy == 'mode':
            for col in self.df.columns:
                mode_val = self.df[col].mode()
                if not mode_val.empty:
                    self.df[col].fillna(mode_val[0], inplace=True)
        elif strategy == 'drop':
            self.df.dropna(inplace=True)
        else:
            raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'.")

    def encode_labels(self, columns: list[str]) -> None:
        """
        Apply label encoding to specific categorical columns.

        Parameters
        ----------
        columns : list of str
            Names of columns to encode.

        Raises
        ------
        ValueError
            If any column does not exist.
        """
        missing = [col for col in columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        
        encoder = LabelEncoder()
        for col in columns:
            self.df[col] = encoder.fit_transform(self.df[col].astype(str))

    def encode_one_hot(self, columns: list[str]) -> None:
        """
        Apply one-hot encoding to specified categorical columns.

        Parameters
        ----------
        columns : list of str
            Names of columns to one-hot encode.
        """
        missing = [col for col in columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)

    def scale_features(self, method: str = 'standard', columns: list[str] = None) -> None:
        """
        Scale numerical features using standard or min-max scaling.

        Parameters
        ----------
        method : {'standard', 'minmax'}
            Scaling method to use.
        columns : list of str, optional
            Subset of columns to scale. If None, scales all numeric columns.

        Raises
        ------
        ValueError
            If the method is not supported.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'.")

        self.df[columns] = self.scaler.fit_transform(self.df[columns])

    def split(self, target_column: str, test_size: float = 0.2, random_state: int = 42):
        """
        Split the dataset into train and test sets.

        Parameters
        ----------
        target_column : str
            Name of the target variable.
        test_size : float
            Proportion of test set.
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        X_train : pd.DataFrame
        X_test : pd.DataFrame
        y_train : pd.Series
        y_test : pd.Series

        Raises
        ------
        ValueError
            If target_column does not exist.
        """
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")
        
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the transformed dataframe.

        Returns
        -------
        pd.DataFrame
            The preprocessed dataset.
        """
        return self.df
