from .classes.dataframe_cleaner import DataFrameCleaner
#from src.data_cleaning import DataFrameCleaner

TOOLS = {
    'cleaner': lambda: DataFrameCleaner(),
}
