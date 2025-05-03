import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils.utils_statistical_learning import (logger, log_method, require_data, 
                                              require_model, require_split)
from abstract_interfaces import AbstractModelManager


class ModelManager(AbstractModelManager):
    """
    Manages the complete machine learning workflow, including:
    data loading, preprocessing, model training, and evaluation.
    """

    def __init__(self, model: BaseEstimator = None, scaler: object = None):
        """
        Initialize the manager with an optional model and scaler.
        
        Parameters:
        - model (BaseEstimator): Scikit-learn compatible model instance.
        - scaler (object): Preprocessing scaler (default: StandardScaler).
        """
        self.model = model
        self.scaler = scaler or StandardScaler()
        self.X = self.y = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    @log_method
    def load_data(self, data_source, target_column: str, loader_func=pd.read_csv, **loader_kwargs):
        """
        Load data from a source and split it into features and target.

        Parameters:
        - data_source: Path or buffer to load the dataset from.
        - target_column (str): Name of the target column.
        - loader_func (function): Function to load the data (default: pd.read_csv).
        - loader_kwargs: Additional keyword arguments passed to the loader function.
        """
        df = loader_func(data_source, **loader_kwargs)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]

    @log_method
    def set_features_and_target(self, X: pd.DataFrame, y: pd.Series):
        """
        Manually set the features and target data.

        Parameters:
        - X (pd.DataFrame): Feature data.
        - y (pd.Series): Target labels.
        """
        if X.shape[0] != len(y):
            raise ValueError("Mismatch between X rows and y length.")
        self.X, self.y = X, y

    @log_method
    @require_data
    def split_data(self, test_size=0.2, random_state=None, stratify=False, **kwargs):
        """
        Split the dataset into training and testing sets.

        Parameters:
        - test_size (float): Proportion of the dataset to include in the test split.
        - random_state (int): Seed used by the random number generator.
        - stratify (bool): Whether to stratify the split using the target variable.
        - kwargs: Additional keyword arguments for `train_test_split`.
        """
        stratify_target = self.y if stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
            **kwargs
        )

    @log_method
    @require_split
    def scale_data(self, scaler=None):
        """
        Scale the training and testing feature sets using a given scaler.

        Parameters:
        - scaler (object): Scaler object implementing `fit_transform` and `transform`.
                           If None, uses the initialized scaler.
        """
        self.scaler = scaler or self.scaler
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    @log_method
    def set_model(self, model: BaseEstimator):
        """
        Set the machine learning model.

        Parameters:
        - model (BaseEstimator): A scikit-learn compatible estimator.
        """
        self.model = model

    @log_method
    @require_split
    @require_model
    def train_model(self, **fit_kwargs):
        """
        Train the model using the training dataset.

        Parameters:
        - fit_kwargs: Additional arguments passed to the model's `fit` method.
        """
        self.model.fit(self.X_train, self.y_train, **fit_kwargs)

    @log_method
    @require_split
    @require_model
    def evaluate_model(self, metrics=None, **predict_kwargs):
        """
        Evaluate the trained model on the test dataset.

        Parameters:
        - metrics (dict): Dictionary of metric names to functions. Defaults to accuracy.
        - predict_kwargs: Additional arguments passed to the model's `predict` method.
        """
        y_pred = self.model.predict(self.X_test, **predict_kwargs)
        metrics = metrics or {"Accuracy": accuracy_score}
        for name, func in metrics.items():
            score = func(self.y_test, y_pred)
            logger.info(f"{name}: {score:.4f}")

