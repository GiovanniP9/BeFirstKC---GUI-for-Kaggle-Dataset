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
    Manages end-to-end ML workflow: loading, preprocessing, training, evaluating.
    """

    def __init__(self, model: BaseEstimator = None, scaler: object = None):
        self.model = model
        self.scaler = scaler or StandardScaler()
        self.X = self.y = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    @log_method
    def load_data(self, data_source, target_column: str, loader_func=pd.read_csv, **loader_kwargs):
        df = loader_func(data_source, **loader_kwargs)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]

    @log_method
    def set_features_and_target(self, X: pd.DataFrame, y: pd.Series):
        if X.shape[0] != len(y):
            raise ValueError("Mismatch between X rows and y length.")
        self.X, self.y = X, y

    @log_method
    @require_data
    def split_data(self, test_size=0.2, random_state=None, stratify=False, **kwargs):
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
        self.scaler = scaler or self.scaler
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    @log_method
    def set_model(self, model: BaseEstimator):
        self.model = model

    @log_method
    @require_split
    @require_model
    def train_model(self, **fit_kwargs):
        self.model.fit(self.X_train, self.y_train, **fit_kwargs)

    @log_method
    @require_split
    @require_model
    def evaluate_model(self, metrics=None, **predict_kwargs):
        y_pred = self.model.predict(self.X_test, **predict_kwargs)
        metrics = metrics or {"Accuracy": accuracy_score}
        for name, func in metrics.items():
            score = func(self.y_test, y_pred)
            logger.info(f"{name}: {score:.4f}")
