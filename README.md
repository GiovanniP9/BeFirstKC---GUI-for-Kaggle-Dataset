# data-analyzer
A simple tool, you should use to analyze data from a pandas dataframe

## **```DataFrameCleaner``` – Clean and Preprocess DataFrames**
### Description:

```DataFrameCleaner``` is a utility class designed to clean and prepare pandas DataFrames by chaining common preprocessing steps such as dropping missing values, filling them, removing duplicates, renaming or dropping columns, etc.
Main Features
- Drop or fill missing values
- Drop duplicate rows
- Rename or remove columns
- Reset index
- Save the cleaned DataFrame to a CSV

### Example Usage:

```python
from cleaner import DataFrameCleaner
import pandas as pd

df = pd.read_csv("data.csv")

clean_df = (DataFrameCleaner(df)
            .drop_missing()
            .fill_missing(strategy="median")
            .drop_duplicates()
            .rename_columns({"old_name": "new_name"})
            .drop_columns(["irrelevant_column"])
            .reset_index()
            .get_df())
```

## **```ModelManager``` – Manage ML Workflow**
### Description:

```ModelManager``` orchestrates a full ML pipeline, including loading data, splitting, scaling, training, and evaluating a scikit-learn compatible model.

### Main Features:
- Load data from a file or set manually
- Train-test split with stratification
- Scale features with custom or default scaler
- Train and evaluate a model with customizable metrics

### Example Usage:

```python
from model_manager import ModelManager
from sklearn.linear_model import LogisticRegression

manager = ModelManager(model=LogisticRegression())
manager.load_data("dataset.csv", target_column="target")
manager.split_data(test_size=0.3, stratify=True)
manager.scale_data()
manager.train_model()
manager.evaluate_model()
```

## **```StatisticalAnalyzer``` – Statistical Summary and Diagnostics**
### Description:

```StatisticalAnalyzer``` provides various statistical analyses for numeric and categorical data, including descriptive statistics, correlation matrices, outlier detection, and distribution plots.

### Main Features:
- Describe numeric data
- Identify and visualize correlations
- Outlier detection via IQR or Z-score
- Summary of missing values and categorical distributions

### Example Usage:

```python
from statistical_analyzer import StatisticalAnalyzer
import pandas as pd

df = pd.read_csv("data.csv")
analyzer = StatisticalAnalyzer(df)

print(analyzer.describe_data())
print(analyzer.skewness_kurtosis())
analyzer.correlation_matrix(plot=True)
analyzer.distribution_plot("age")
outliers = analyzer.outlier_summary("salary", method="zscore")
```

## **```EDA``` – Exploratory Data Analysis**
### Description:

The ```EDA``` class provides tools for performing exploratory data analysis (EDA) through visualization and statistical summaries of both numerical and categorical features.

### Main Features:
- Print dataset shape, info, nulls, and statistics
- Plot histograms, boxplots, and correlation heatmaps
- Visualize relationships between features and the target variable

### Example Usage:

```python
from eda import EDA
import pandas as pd

df = pd.read_csv("data.csv")
eda = EDA(df)

eda.check_basic_info()
eda.plot_histograms()
eda.plot_boxplots()
eda.plot_correlation_matrix()
eda.plot_target_relations(target="price")
```

## **```Menu``` – CLI Method Invoker**

### Description:

```Menu``` allows for dynamic and interactive exploration of any class instance via the command line. It lists all public methods and prompts the user to provide necessary inputs to invoke them.

### Main Features:
- Introspects public methods using inspect
- Casts user inputs to the correct types
- Supports interactive workflows for classes with methods

### Example Usage:

```python
from menu import Menu
from eda import EDA
import pandas as pd

df = pd.read_csv("data.csv")
eda = EDA(df)
menu = Menu(eda)
menu.select_and_execute()
```