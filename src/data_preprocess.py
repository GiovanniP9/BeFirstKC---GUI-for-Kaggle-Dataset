import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


class Preprocessor:
    """
    A class for preprocessing tasks before model training.
    Includes handling missing values, encoding categorical variables,
    scaling numerical features, and splitting datasets.
    """

    def __init__(self, dataframe):
        """
        Initialize the Preprocessor with a pandas DataFrame.
        
        Parameters:
        dataframe (pd.DataFrame): The dataset to preprocess.
        """
        self.df = dataframe.copy()
        self.scaler = None

    def fill_missing(self, strategy='mean'):
        """
        Fill missing values in the dataset.
        
        Parameters:
        strategy (str): The strategy to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'.
        """
        if strategy == 'mean':
            self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
        elif strategy == 'median':
            self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            self.df.dropna(inplace=True)
        else:
            raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")

    def encode_labels(self, columns):
        """
        Apply label encoding to the specified categorical columns.
        
        Parameters:
        columns (list): List of column names to encode.
        """
        encoder = LabelEncoder()
        for col in columns:
            self.df[col] = encoder.fit_transform(self.df[col])

    def encode_one_hot(self, columns):
        """
        Apply one-hot encoding to the specified categorical columns.
        
        Parameters:
        columns (list): List of column names to one-hot encode.
        """
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)

    def scale_features(self, method='standard', columns=None):
        """
        Scale numerical features using StandardScaler or MinMaxScaler.
        
        Parameters:
        method (str): Scaling method, 'standard' or 'minmax'.
        columns (list): List of columns to scale. If None, all numeric columns are scaled.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")

        self.df[columns] = self.scaler.fit_transform(self.df[columns])

    def split(self, target_column, test_size=0.2, random_state=42):
        """
        Split the dataset into training and testing sets.
        
        Parameters:
        target_column (str): The name of the target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
        Returns:
        X_train, X_test, y_train, y_test: Split feature and target datasets.
        """
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def get_processed_data(self):
        """
        Return the preprocessed dataframe.
        
        Returns:
        pd.DataFrame: The transformed dataset.
        """
        return self.df


"""
========== USE CASE ==========

df = pd.read_csv('dataset.csv')
prep = Preprocessor(df)

prep.fill_missing(strategy='mean')
prep.encode_one_hot(columns=['gender', 'region'])
prep.scale_features(method='standard')
X_train, X_test, y_train, y_test = prep.split(target_column='target')

"""