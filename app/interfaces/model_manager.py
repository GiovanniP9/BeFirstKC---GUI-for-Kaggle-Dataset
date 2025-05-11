import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from app.utils.utils_model_manager import (
    logger, log_method, require_data, 
    require_model, require_split
)
from app.abstract.abstract_interfaces import AbstractModelManager


class ModelManager(AbstractModelManager):
    """
    Manages the full machine learning workflow: loading data, 
    preprocessing, model training, and evaluation.
    """

    def __init__(self, model: BaseEstimator = None, scaler: object = None):
        """
        Initialize the ModelManager with optional model and scaler.

        Parameters
        ----------
        model : BaseEstimator, optional
            A scikit-learn compatible estimator.
        scaler : object, optional
            A scaler implementing `fit_transform` and `transform`.
            Defaults to `StandardScaler`.
        """
        self.model = model
        self.scaler = scaler or StandardScaler()
        self.X = self.y = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    @log_method
    def load_data(self, data_source, target_column: str, loader_func=pd.read_csv, **loader_kwargs) -> None:
        """
        Load a dataset and separate it into features and target.

        Parameters
        ----------
        data_source : str or buffer
            Path or buffer to load the dataset from.
        target_column : str
            Name of the target column.
        loader_func : callable, optional
            Function to load data (default: `pd.read_csv`).
        loader_kwargs : dict
            Additional keyword arguments for `loader_func`.

        Raises
        ------
        ValueError
            If the target column is not found in the dataset.
        """
        df = loader_func(data_source, **loader_kwargs)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]

    @log_method
    def set_features_and_target(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Manually set the feature matrix and target vector.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.
        y : pd.Series
            Target labels.

        Raises
        ------
        ValueError
            If the number of samples in X and y do not match.
        """
        if X.shape[0] != len(y):
            raise ValueError("Mismatch between X rows and y length.")
        self.X, self.y = X, y

    @log_method
    @require_data
    def split_data(self, test_size=0.2, random_state=None, stratify=False, **kwargs) -> None:
        """
        Split the data into training and test sets.

        Parameters
        ----------
        test_size : float, optional
            Proportion of the dataset to use for testing.
        random_state : int, optional
            Random seed for reproducibility.
        stratify : bool, optional
            Whether to stratify by target variable.
        kwargs : dict
            Additional parameters for `train_test_split`.
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
    def scale_data(self, scaler=None) -> None:
        """
        Apply scaling to the training and testing feature data.

        Parameters
        ----------
        scaler : object, optional
            A scikit-learn scaler. If None, uses the existing scaler.
        """
        self.scaler = scaler or self.scaler
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    @log_method
    def set_model(self, model: BaseEstimator) -> None:
        """
        Set the estimator model to be trained.

        Parameters
        ----------
        model : BaseEstimator
            A scikit-learn compatible estimator.
        """
        self.model = model

    @log_method
    @require_split
    @require_model
    def train_model(self, **fit_kwargs) -> None:
        """
        Train the model using the training dataset.

        Parameters
        ----------
        fit_kwargs : dict
            Additional keyword arguments passed to `model.fit`.
        """
        self.model.fit(self.X_train, self.y_train, **fit_kwargs)

    @log_method
    @require_split
    @require_model
    def evaluate_model(self, metrics=None, **predict_kwargs) -> dict:
        """
        Evaluate the trained model on the test dataset.

        Parameters
        ----------
        metrics : dict, optional
            Dictionary of metric name to callable (e.g., `accuracy_score`).
            If None, defaults to accuracy.
        predict_kwargs : dict
            Additional arguments passed to `model.predict`.

        Returns
        -------
        dict
            A dictionary containing metric names and their computed values.
        """
        y_pred = self.model.predict(self.X_test, **predict_kwargs)
        metrics = metrics or {"Accuracy": accuracy_score}
        results = {}
        for name, func in metrics.items():
            score = func(self.y_test, y_pred)
            logger.info(f"{name}: {score:.4f}")
            results[name] = score
        return results
