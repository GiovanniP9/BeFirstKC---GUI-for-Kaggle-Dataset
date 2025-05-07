import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# file paths
train_path = 'csv\\train.csv'
test_path = 'csv\\test.csv'
# read data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# check data
print("train_data head:", train_data.head())
print("train_data shape:", train_data.shape)
print("train data describer:", train_data.describe())

#copia del train data
data = train_data.copy()

#colonne che vogliamo fillare i missing values
cols_to_fill = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", "Waterproof"]

# fill temporaneo per poter usare label encoder
data.fillna(-999, inplace=True)

# label encoding per le colonne categoriche
encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    # save the encoder for later use
    encoders[col] = le

# fillare i missing values con random forest
for col in cols_to_fill:
    if train_data[col].isnull().sum() == 0:
        continue # skip if no missing values
    
    print(f"Filling missing values for {col}...")
    
    X = data.drop(columns=[col])
    y = train_data[col]
    
    X_train = X[y.notnull()] # selezione delle righe con valori non nulli
    y_train = y[y.notnull()] # selezione delle righe con valori non nulli
    X_test = X[y.isnull()] # selezione delle righe con valori nulli
    
    # selezione del modello basato sul tipo di dato
    if y.dtype == 'object':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        y_le = LabelEncoder() # label encoding per la variabile target
        y_train_encoded = y_le.fit_transform(y_train.astype(str))
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)
        train_data.loc[y.isnull(), col] = y_le.inverse_transform(y_pred)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_data.loc[y.isnull(), col] = y_pred

print("Missing values filled.")
       
print(data.isnull().sum())
# check if there are still missing values

y = train_data['Price']

features = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof' ]

test_data_cleaned = test_data.dropna(axis=0) # remove rows with missing values

X = train_data[features]
X_test = test_data[features]

X_encoded = pd.get_dummies(X)
X_test_encoded = pd.get_dummies(X_test)

#split the data into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, random_state=42)

# train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
print(f"MAE: {mae:.2f}")
#root mean squared error

rmse = np.sqrt(mean_absolute_error(y_val, preds))
print(f"RMSE: {rmse:.2f}")

test_preds = model.predict(X_test_encoded.loc[test_data_cleaned.index])
submission = pd.DataFrame({'id': test_data_cleaned['id'], 'Price': test_preds})
submission.to_csv('submission.csv', index=False)
print("Submission file created.")