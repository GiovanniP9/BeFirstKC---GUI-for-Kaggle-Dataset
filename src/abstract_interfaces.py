from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator


class AbstractDataFrameCleaner(ABC):
    """
    Interface for a pandas dataframe cleaner.
    """
    @abstractmethod
    def drop_missing(self, axis=0, how='any', thresh=None, subset=None): 
        pass

    @abstractmethod
    def fill_missing(self, strategy='mean', columns=None): 
        pass

    @abstractmethod
    def drop_duplicates(self, subset=None, keep='first'): 
        pass

    @abstractmethod
    def rename_columns(self, rename_dict): 
        pass

    @abstractmethod
    def drop_columns(self, columns): 
        pass

    @abstractmethod
    def reset_index(self, drop=True): 
        pass

    @abstractmethod
    def get_df(self): 
        pass

    @abstractmethod
    def to_csv(self, path, index=False): 
        pass


class AbstractEDA(ABC):
    """
    Abstract interface for exploration data analysis.
    """
    @abstractmethod
    def check_basic_info(self): 
        pass

    @abstractmethod
    def summarize_numerical(self): 
        pass

    @abstractmethod
    def summarize_categorical(self): 
        pass

    @abstractmethod
    def plot_histograms(self, columns=None): 
        pass

    @abstractmethod
    def plot_boxplots(self, columns=None): 
        pass

    @abstractmethod
    def plot_correlation_matrix(self): 
        pass

    @abstractmethod
    def plot_target_relations(self, target): 
        pass
    
class AbstractStatisticalAnalyser(ABC):
    """
    Abstract interface for statistical data analysis.
    """
    @abstractmethod
    def describe_data(self):
        pass
    
    @abstractmethod
    def missing_values_summary(self):
        pass
    
    @abstractmethod
    def correlation_matrix(self, method='pearson', plot=False):
        pass
    
    @abstractmethod
    def distribution_plot(self, column: str):
        pass
        
    @abstractmethod
    def skewness_kurtosis(self):
        pass
    
    @abstractmethod
    def outlier_summary(self, column: str, method='iqr'):
        pass
    
    @abstractmethod
    def categorical_summary(self):
        pass
    

class AbstractPreprocessor(ABC):
    """
    Abstract interface for data preprocessing before model training.
    """
    @abstractmethod
    def fill_missing(self, strategy='mean'):
        pass

    @abstractmethod
    def encode_labels(self, columns):
        pass

    @abstractmethod
    def encode_one_hot(self, columns):
        pass

    @abstractmethod
    def scale_features(self, method='standard', columns=None):
        pass

    @abstractmethod
    def split(self, target_column, test_size=0.2, random_state=42):
        pass

    @abstractmethod
    def get_processed_data(self):
        pass


class AbstractModelManager(ABC):
    """
    Abstract interface for ML model management.
    """

    @abstractmethod
    def load_data(self, data_source, target_column, loader_func=pd.read_csv, **loader_kwargs):
        pass

    @abstractmethod
    def set_features_and_target(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def split_data(self, test_size: float, random_state: int, stratify: bool, **kwargs):
        pass

    @abstractmethod
    def scale_data(self, scaler=None):
        pass

    @abstractmethod
    def set_model(self, model: BaseEstimator):
        pass

    @abstractmethod
    def train_model(self, **fit_kwargs):
        pass

    @abstractmethod
    def evaluate_model(self, metrics=None, **predict_kwargs):
        pass
    
    
class AbstractStatisticalAnalyzer(ABC):
    """
    Interface for statistical analysis of a pandas DataFrame.
    """

    @abstractmethod
    def describe_data(self):
        pass

    @abstractmethod
    def missing_values_summary(self):
        pass

    @abstractmethod
    def correlation_matrix(self, method='pearson', plot=False):
        pass

    @abstractmethod
    def distribution_plot(self, column: str):
        pass

    @abstractmethod
    def skewness_kurtosis(self):
        pass

    @abstractmethod
    def outlier_summary(self, column: str, method='iqr'):
        pass

    @abstractmethod
    def categorical_summary(self):
        pass