import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# --- Funzioni per pulizia e imputazione dei dati ---

def load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill):
    if os.path.exists(cleaned_path):
        print("Caricamento del dataset pulito...")
        return pd.read_csv(cleaned_path)
    
    print("Dataset non pulito trovato. Eseguo imputazione...")
    raw_data = pd.read_csv(train_path)
    data_encoded, encoders = encode_categoricals(raw_data.copy())

    cleaned_data = impute_missing_values(raw_data, data_encoded, cols_to_fill)
    cleaned_data.to_csv(cleaned_path, index=False)
    print(f"Dataset pulito salvato in: {cleaned_path}")
    return cleaned_data

def encode_categoricals(data):
    encoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna("missing")
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
    return data, encoders

def impute_missing_values(original_data, encoded_data, cols_to_fill):
    for col in cols_to_fill:
        if original_data[col].isnull().sum() == 0:
            continue

        print(f"Imputazione per la colonna: {col}")
        y = original_data[col]
        X = encoded_data.drop(columns=[col])

        X_train = X[y.notnull()]
        y_train = y[y.notnull()]
        X_test = X[y.isnull()]

        if y.dtype == 'object':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            y_le = LabelEncoder()
            y_train_encoded = y_le.fit_transform(y_train.astype(str))
            model.fit(X_train, y_train_encoded)
            y_pred = model.predict(X_test)
            original_data.loc[y.isnull(), col] = y_le.inverse_transform(y_pred)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            original_data.loc[y.isnull(), col] = y_pred

    return original_data

# --- Funzioni per preparazione dei dati e addestramento ---

def prepare_features(train_data, test_data, features):
    X = train_data[features]
    y = train_data['Price']
    X_test = test_data[features].dropna()
    test_ids = test_data.loc[X_test.index, 'id']

    X_all = pd.concat([X, X_test], keys=['train', 'test'])
    X_all_encoded = pd.get_dummies(X_all)

    X_encoded = X_all_encoded.xs('train')
    X_test_encoded = X_all_encoded.xs('test')
    
    return X_encoded, X_test_encoded, y, test_ids

def run_optuna_tuning(X_train, X_val, y_train, y_val, n_trials=50):
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
        return np.sqrt(mean_squared_error(y_val, preds))
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print("🔧 Migliori parametri trovati:", study.best_params)
    return study.best_params

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return model

# --- Funzione per creazione submission ---

def create_submission(model, X_test_encoded, test_ids, output_path='submission.csv'):
    preds = model.predict(X_test_encoded)
    submission = pd.DataFrame({'id': test_ids, 'Price': preds})
    submission.to_csv(output_path, index=False)
    print(f"File di sottomissione salvato in: {output_path}")

# --- MAIN ---

if __name__ == "__main__":
    train_path = 'csv/train.csv'
    test_path = 'csv/test.csv'
    cleaned_path = 'csv/train_cleaned.csv'
    cols_to_fill = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", 
                    "Waterproof", "Style", "Color", "Weight Capacity (kg)"]
    features = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", "Waterproof"]

    train_data = load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill)
    test_data = pd.read_csv(test_path)

    X_encoded, X_test_encoded, y, test_ids = prepare_features(train_data, test_data, features)
    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, random_state=42)

    best_params = run_optuna_tuning(X_train, X_val, y_train, y_val, n_trials=50)
    model = train_final_model(X_train, y_train, X_val, y_val, best_params)
    create_submission(model, X_test_encoded, test_ids)
