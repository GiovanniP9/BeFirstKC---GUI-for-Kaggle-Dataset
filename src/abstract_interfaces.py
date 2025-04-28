from abc import ABC, abstractmethod


class AbstractDataFrameCleaner(ABC):
    
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
    def get_df(self):
        pass
    
    @abstractmethod
    def reset_index(self, drop=True):
        pass
    
    @abstractmethod
    def to_csv(self, path, index=False):
        pass
