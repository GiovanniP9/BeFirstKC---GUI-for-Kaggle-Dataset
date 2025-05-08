import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# --- 1) Pulizia e imputazione dei dati ---
# Funzione per caricare o preparare un dataset pulito
def load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill):
    if os.path.exists(cleaned_path):
        # Se il dataset pulito esiste, lo carica
        print("Caricamento del dataset pulito...")
        return pd.read_csv(cleaned_path)
    # Altrimenti, esegue la pulizia e l'imputazione
    print("Dataset non pulito trovato. Eseguo imputazione...")
    raw_df = pd.read_csv(train_path)
    encoded_df, _ = encode_categoricals(raw_df.copy())
    cleaned_df    = impute_missing_values(raw_df, encoded_df, cols_to_fill)
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Dataset pulito salvato in: {cleaned_path}")
    return cleaned_df

# Funzione per codificare le colonne categoriche
def encode_categoricals(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("missing")  # Riempie i valori mancanti con "missing"
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Codifica le colonne categoriche
    return df, None

# Funzione per imputare i valori mancanti
def impute_missing_values(orig_df, enc_df, cols_to_fill):
    for col in cols_to_fill:
        if orig_df[col].isnull().sum() == 0:
            continue  # Salta le colonne senza valori mancanti
        print(f"Imputazione colonna: {col}")
        y = orig_df[col]
        X = enc_df.drop(columns=[col])
        X_known, X_missing = X[y.notnull()], X[y.isnull()]
        y_known = y[y.notnull()]
        if orig_df[col].dtype == 'object':
            # Usa RandomForestClassifier per colonne categoriche
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            le  = LabelEncoder()
            y_enc = le.fit_transform(y_known.astype(str))
            clf.fit(X_known, y_enc)
            y_pred = le.inverse_transform(clf.predict(X_missing))
        else:
            # Usa RandomForestRegressor per colonne numeriche
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_known, y_known)
            y_pred = reg.predict(X_missing)
        orig_df.loc[y.isnull(), col] = y_pred  # Sostituisce i valori mancanti con le predizioni
    return orig_df

# --- 2) Preparazione, encoding e SCALING delle features ---
# Funzione per preparare e scalare le features
def prepare_and_scale_features(train_df, test_df, features):
    X_train_raw = train_df[features]
    y_train     = train_df['Price']  # Target
    X_test_raw  = test_df[features]
    test_ids    = test_df['id']  # Identificatori per il test set

    # Combina i dataset per un encoding coerente
    X_all         = pd.concat([X_train_raw, X_test_raw], keys=['train', 'test'])
    X_all_dummy   = pd.get_dummies(X_all)  # One-hot encoding
    X_dummy_train = X_all_dummy.xs('train')
    X_dummy_test  = X_all_dummy.xs('test')

    # Divide il dataset di training in training e validation set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dummy_train, y_train, test_size=0.2, random_state=42
    )

    # Applica lo scaling standard
    scaler = StandardScaler()
    X_train_scaled_model = scaler.fit_transform(X_tr)
    X_val_scaled_model   = scaler.transform(X_val)
    X_test_scaled_model  = scaler.transform(X_dummy_test)
    return X_train_scaled_model, X_val_scaled_model, y_tr, y_val, X_test_scaled_model, test_ids

# --- 3) Funzioni di tuning per XGB e LightGBM ---
# Funzione per il tuning di XGBoost
def tune_xgb(X_train, X_val, y_train, y_val, n_trials=50):
    def obj(trial):
        # Definizione dello spazio dei parametri
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
        }
        # Addestra il modello con i parametri suggeriti
        model = XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, pred))  # Ritorna l'RMSE
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=n_trials)
    return study.best_params

# Funzione per il tuning di LightGBM
def tune_lgbm(X_train, X_val, y_train, y_val, n_trials=50):
    def obj(trial):
        # Definizione dello spazio dei parametri
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
        }
        # Addestra il modello con i parametri suggeriti
        model = LGBMRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, pred))  # Ritorna l'RMSE
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=n_trials)
    return study.best_params

# --- 4) Valutazione con parametri ottimizzati ---
# Funzione per valutare i modelli
def evaluate_models(models, X_train, X_val, y_train, y_val):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)  # Addestra il modello
        preds = model.predict(X_val)  # Predice sul validation set
        mae = mean_absolute_error(y_val, preds)  # Calcola MAE
        rmse = np.sqrt(mean_squared_error(y_val, preds))  # Calcola RMSE
        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}")
        results[name] = {'MAE': mae, 'RMSE': rmse}
    return results

# --- 5) Creazione submission per il modello migliore ---
# Funzione per creare il file di submission
def create_submission(model, X_test, test_ids, output_path='submission.csv'):
    preds = model.predict(X_test)  # Predice sul test set
    pd.DataFrame({'id': test_ids, 'Price': preds}).to_csv(output_path, index=False)
    print(f"Submission salvata per il modello migliore in: {output_path}")

# --- MAIN ---
if __name__ == '__main__':
    # Percorsi dei file
    train_path, test_path, cleaned_path = 'csv/train.csv', 'csv/test.csv', 'csv/train_cleaned.csv'
    # Colonne da imputare
    cols_to_fill = ["Brand","Material","Size","Compartments",
                    "Laptop Compartment","Waterproof","Style",
                    "Color","Weight Capacity (kg)"]
    features = cols_to_fill.copy()

    # Caricamento e preparazione dei dati
    train_df = load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill)
    test_df  = pd.read_csv(test_path)
    X_train, X_val, y_train, y_val, X_test, test_ids = prepare_and_scale_features(
        train_df, test_df, features
    )

    # Tuning per XGB e LightGBM
    best_xgb = tune_xgb(X_train, X_val, y_train, y_val)
    best_lgb = tune_lgbm(X_train, X_val, y_train, y_val)

    # Costruzione dei modelli con parametri ottimizzati e modelli di riferimento
    models = {
        'XGB': XGBRegressor(**best_xgb, random_state=42),
        'LightGBM': LGBMRegressor(**best_lgb, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Valutazione e selezione del migliore
    results = evaluate_models(models, X_train, X_val, y_train, y_val)
    best_name = min(results, key=lambda k: results[k]['RMSE'])  # Seleziona il modello con RMSE minimo
    print(f"Best model: {best_name}")
    best_model = models[best_name]

    # Creazione submission solo per il migliore
    create_submission(best_model, X_test, test_ids)