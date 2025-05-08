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
# Carica il CSV pulito se esiste, altrimenti esegue imputazione e salva il dataset pulito

def load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill):
    if os.path.exists(cleaned_path):
        print("Caricamento del dataset pulito...")
        df = pd.read_csv(cleaned_path)
    else:
        print("Dataset non pulito trovato. Eseguo imputazione...")
        raw_df = pd.read_csv(train_path)
        encoded_df, _ = encode_categoricals(raw_df.copy())
        df = impute_missing_values(raw_df, encoded_df, cols_to_fill)
        df.to_csv(cleaned_path, index=False)
        print(f"Dataset pulito salvato in: {cleaned_path}")
    # Creazione di feature aggiuntive basate sul target Price (solo su train)
    if 'Price' in df.columns:
        df['price_per_kg'] = df['Price'] / df['Weight Capacity (kg)']
        df['price_per_compartment'] = df['Price'] / df['Compartments']
        df['capacity_per_compartment'] = df['Weight Capacity (kg)'] / df['Compartments']
        df['price_per_kg_per_compartment'] = df['price_per_kg'] / df['Compartments']
    else:
        # Se per errore Price non è presente, inizializza a zero
        df['price_per_kg'] = 0
        df['price_per_compartment'] = 0
        df['capacity_per_compartment'] = df['Weight Capacity (kg)'] / df['Compartments']
        df['price_per_kg_per_compartment'] = 0
    return df


def encode_categoricals(df):
    # Trasforma categoriche in numeriche con LabelEncoder
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df, None


def impute_missing_values(orig_df, enc_df, cols_to_fill):
    # Imputa valori mancanti con RandomForest
    for col in cols_to_fill:
        missing_count = orig_df[col].isnull().sum()
        if missing_count == 0:
            continue
        print(f"Imputazione colonna: {col} (mancanti: {missing_count})")
        y = orig_df[col]
        X = enc_df.drop(columns=[col])
        X_known, X_missing = X[y.notnull()], X[y.isnull()]
        y_known = y[y.notnull()]
        if orig_df[col].dtype == 'object':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            le  = LabelEncoder()
            y_enc = le.fit_transform(y_known.astype(str))
            clf.fit(X_known, y_enc)
            y_pred = le.inverse_transform(clf.predict(X_missing))
        else:
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_known, y_known)
            y_pred = reg.predict(X_missing)
        orig_df.loc[y.isnull(), col] = y_pred
    return orig_df

# --- 2) Preparazione, encoding e SCALING delle feature ---
# Gestisce missing feature nel test riempiendo a zero

def prepare_and_scale_features(train_df, test_df, features):
    # Estrai train raw e target
    X_train_raw = train_df[features]
    y_train     = train_df['Price']
    # Per il test, reindicizza le colonne mancanti a zero
    X_test_raw  = test_df.reindex(columns=features, fill_value=0)
    test_ids    = test_df['id']

    # One-hot encoding congiunto per coerenza
    X_all       = pd.concat([X_train_raw, X_test_raw], keys=['train', 'test'])
    X_all_dummy = pd.get_dummies(X_all)
    X_dummy_train = X_all_dummy.xs('train')
    X_dummy_test  = X_all_dummy.xs('test')

    # Split train/validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dummy_train, y_train, test_size=0.2, random_state=42
    )

    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_tr)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_dummy_test)

    return X_train_scaled, X_val_scaled, y_tr, y_val, X_test_scaled, test_ids

# --- 3) Funzioni di tuning per XGBoost e LightGBM ---

def tune_xgb(X_train, X_val, y_train, y_val, n_trials=50):
    study = optuna.create_study(direction='minimize')
    def obj(trial):
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
        model = XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    study.optimize(obj, n_trials=n_trials)
    print("Tuning XGB completato.")
    return study.best_params


def tune_lgbm(X_train, X_val, y_train, y_val, n_trials=50):
    study = optuna.create_study(direction='minimize')
    def obj(trial):
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
        model = LGBMRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    study.optimize(obj, n_trials=n_trials)
    print("Tuning LGBM completato.")
    return study.best_params

# --- 4) Valutazione dei modelli ---
def evaluate_models(models, X_train, X_val, y_train, y_val):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        results[name] = {'MAE': mae, 'RMSE': rmse}
    return results

# --- 5) Creazione submission per il migliore modello ---
def create_submission(model, X_test, test_ids, output_path='submission.csv'):
    preds = model.predict(X_test)
    pd.DataFrame({'id': test_ids, 'Price': preds}).to_csv(output_path, index=False)
    print(f"Submission salvata per il modello migliore in: {output_path}")

# --- MAIN ---
if __name__ == '__main__':
    # Percorsi e colonne da imputare
    train_path, test_path = 'csv/train.csv', 'csv/test.csv'
    cleaned_path = 'csv/train_cleaned.csv'
    cols_to_fill = [
        "Brand","Material","Size","Compartments",
        "Laptop Compartment","Waterproof","Style",
        "Color","Weight Capacity (kg)"
    ]
    # Caricamento e pulizia del train
    train_df = load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill)
    # Caricamento del test
    test_df  = pd.read_csv(test_path)
    
    # Definizione delle feature, incluse le nuove interaction feature
    feature_cols = cols_to_fill + [
        'price_per_kg','price_per_compartment',
        'capacity_per_compartment','price_per_kg_per_compartment'
    ]
    
    # Preparazione e scaling delle feature per train e test
    X_train, X_val, y_train, y_val, X_test, test_ids = prepare_and_scale_features(
        train_df, test_df, feature_cols
    )
    
    # Ottimizzazione dei parametri su XGB e LGBM
    best_xgb = tune_xgb(X_train, X_val, y_train, y_val)
    best_lgb = tune_lgbm(X_train, X_val, y_train, y_val)
    
    # Definizione dei modelli (XGB e LGBM con tuning, RF e GB con default)
    models = {
        'XGB': XGBRegressor(**best_xgb, random_state=42),
        'LightGBM': LGBMRegressor(**best_lgb, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Valutazione e selezione del migliore modello
    results = evaluate_models(models, X_train, X_val, y_train, y_val)
    best_name = min(results, key=lambda k: results[k]['RMSE'])
    print(f"Miglior modello selezionato: {best_name}")
    best_model = models[best_name]

    # Creazione della submission per il modello migliore
    create_submission(best_model, X_test, test_ids)
