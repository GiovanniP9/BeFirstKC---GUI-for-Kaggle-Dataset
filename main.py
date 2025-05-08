import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# file paths
train_path = 'csv\\train.csv'
test_path = 'csv\\test.csv'

# read data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# copia per modifiche
data = train_data.copy()
cols_to_fill = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", "Waterproof", "Style", "Color", "Weight Capacity (kg)"]

# Encoding delle categoriche con placeholder per i NaN
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].fillna("missing")

encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Imputazione dei missing values
for col in cols_to_fill:
    if train_data[col].isnull().sum() == 0:
        continue
    
    print(f"Filling missing values in column: {col}")

    y = train_data[col]
    X = data.drop(columns=[col])

    X_train = X[y.notnull()]
    y_train = y[y.notnull()]
    X_test = X[y.isnull()]

    if y.dtype == 'object':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        y_le = LabelEncoder()
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

# Preparo dati per modello
y = train_data['Price']
features = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", "Waterproof"]
X = train_data[features]
X_test = test_data[features]

X_test_cleaned = X_test.dropna()
test_ids = test_data.loc[X_test_cleaned.index, 'id']

X_all = pd.concat([X, X_test_cleaned], keys=['train', 'test'])
X_all_encoded = pd.get_dummies(X_all)
X_encoded = X_all_encoded.xs('train')
X_test_encoded = X_all_encoded.xs('test')

X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, random_state=42)

# ----------- OPTUNA STARTS HERE -----------

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse  # Optuna minimizza la metrica

# ottimizzazione
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# migliori iperparametri
print("Best params:", study.best_params)

# modello finale
best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)

# valutazione
preds = best_model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# predizione su test
test_preds = best_model.predict(X_test_encoded)

# salvataggio
submission = pd.DataFrame({'id': test_ids, 'Price': test_preds})
submission.to_csv('submission.csv', index=False)
print("Submission file created.")
