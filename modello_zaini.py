import os
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Paths per salvataggio
TRAIN_PATH = 'csv/train.csv'
TEST_PATH = 'csv/test.csv'
CLEAN_TRAIN_CSV = 'csv/train_cleaned.csv'
CLEAN_TEST_CSV = 'csv/test_cleaned.csv'
ENCODERS_PKL = 'models/encoders.pkl'

# --- STEP 1: Pulizia - Encoding categoriali ---
def encode_categoricals(df, encoders=None):
    out_encoders = {} if encoders is None else encoders.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("missing").astype(str)
        if encoders is None:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            out_encoders[col] = le
            print(f"[encode] Colonna '{col}' codificata su train; classi: {len(le.classes_)}")
        else:
            df[col] = out_encoders[col].transform(df[col])
            print(f"[encode] Colonna '{col}' trasformata su test")
    return df, out_encoders

# --- STEP 2: Pulizia - Imputazione valori mancanti train ---
def impute_missing_values(orig_df, enc_df, cols_to_fill):
    imputers = {}
    exclude = ['id', 'Price']

    # 2.1 Train dei modelli su righe non-null
    for col in cols_to_fill:
        y = orig_df[col]
        mask_notnull = y.notnull()
        X_full = enc_df.drop(columns=[c for c in exclude + [col] if c in enc_df.columns]).loc[mask_notnull]
        y_notnull = y[mask_notnull]
        if orig_df[col].dtype == 'object':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            le_y = LabelEncoder().fit(y_notnull.astype(str))
            clf.fit(X_full, le_y.transform(y_notnull.astype(str)))
            imputers[col] = ('clf', clf, le_y)
            print(f"[impute-train] Classifier per '{col}'; classi: {len(le_y.classes_)} su {mask_notnull.sum()} campioni")
        else:
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_full, y_notnull)
            imputers[col] = ('reg', reg, None)
            print(f"[impute-train] Regressor per '{col}' su {mask_notnull.sum()} campioni")

    # 2.2 Fill dei NaN
    for col, (kind, model, le_y) in imputers.items():
        mask = orig_df[col].isnull()
        if not mask.any():
            continue
        X_missing = enc_df.loc[mask].drop(columns=[c for c in exclude + [col] if c in enc_df.columns])
        if kind == 'clf':
            filled = le_y.inverse_transform(model.predict(X_missing))
            orig_df.loc[mask, col] = filled
            print(f"[impute-fill] '{col}': {mask.sum()} NaN fillati (clf)")
        else:
            filled = model.predict(X_missing)
            orig_df.loc[mask, col] = filled
            print(f"[impute-fill] '{col}': {mask.sum()} NaN fillati (reg)")
    return orig_df, imputers

# --- STEP 3: Caching clean train (solo encoder) ---
def load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill):
    # Caricamento se esistono train pulito e encoder
    if os.path.exists(cleaned_path) and os.path.exists(ENCODERS_PKL):
        print("Carico train clean e encoder...")
        clean_df = pd.read_csv(cleaned_path)
        encoders = joblib.load(ENCODERS_PKL)
        # Se esiste già test_cleaned, salto l'addestramento degli imputers
        if os.path.exists(CLEAN_TEST_CSV):
            print("Test clean già presente, imputers non ricostruiti")
            imputers = {}
        else:
            # Rigenero imputers da clean_df solo se serve imputare test
            enc_df, _ = encode_categoricals(clean_df.copy(), encoders)
            _, imputers = impute_missing_values(clean_df.copy(), enc_df, cols_to_fill)
        return clean_df, encoders, imputers

    # Altrimenti pulisco e creo encoder e imputers da zero
    print("Pulizia train da zero...")
    raw_df = pd.read_csv(train_path)
    enc_df, encoders = encode_categoricals(raw_df.copy())
    clean_df, imputers = impute_missing_values(raw_df.copy(), enc_df, cols_to_fill)
    # Salvo solo CSV pulito e encoder
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    clean_df.to_csv(cleaned_path, index=False)
    os.makedirs(os.path.dirname(ENCODERS_PKL), exist_ok=True)
    joblib.dump(encoders, ENCODERS_PKL)
    print(f"Salvati: {cleaned_path}, {ENCODERS_PKL}")
    return clean_df, encoders, imputers

# --- STEP 4: Caching clean test ---
def load_or_prepare_clean_test(test_path, clean_test_path, encoders, imputers, cols_to_fill):
    if os.path.exists(clean_test_path):
        print("Carico test clean...")
        return pd.read_csv(clean_test_path)
    print("Pulizia test da zero...")
    test_df = pd.read_csv(test_path)
    test_df = impute_test_values(test_df, encoders, imputers, cols_to_fill)
    os.makedirs(os.path.dirname(clean_test_path), exist_ok=True)
    test_df.to_csv(clean_test_path, index=False)
    print(f"Salvato test clean: {clean_test_path}")
    return test_df

# --- STEP 5: Imputazione test set ---
def impute_test_values(test_df, encoders, imputers, cols_to_fill):
    enc_test_df = test_df.copy()
    for col, le in encoders.items():
        enc_test_df[col] = enc_test_df[col].fillna("missing").astype(str)
        enc_test_df[col] = le.transform(enc_test_df[col])
        print(f"[encode-test] '{col}' trasformata")
    exclude = ['id', 'Price']
    for col, (kind, model, le_y) in imputers.items():
        mask = test_df[col].isnull()
        if not mask.any(): continue
        X_miss = enc_test_df.loc[mask].drop(columns=[c for c in exclude + [col] if c in enc_test_df.columns])
        if kind == 'clf':
            test_df.loc[mask, col] = le_y.inverse_transform(model.predict(X_miss))
            print(f"[impute-test] '{col}': {mask.sum()} NaN fillati (clf)")
        else:
            test_df.loc[mask, col] = model.predict(X_miss)
            print(f"[impute-test] '{col}': {mask.sum()} NaN fillati (reg)")
    return test_df

# --- STEP 6: Preprocessing & scaling ---
def prepare_and_scale_features(train_df, test_df, features):
    X_tr_raw = train_df[features]
    y_train = train_df['Price']
    X_te_raw = test_df[features]
    ids = test_df['id']
    X_all = pd.concat([X_tr_raw, X_te_raw], keys=['train','test'])
    X_all_d = pd.get_dummies(X_all)
    X_tr_d = X_all_d.xs('train')
    X_te_d = X_all_d.xs('test')
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr_d, y_train, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_te_d)
    return X_train_s, X_val_s, y_tr, y_val, X_test_s, ids

# --- STEP 7: Hyperparameter tuning ---
def tune_xgb(X_train, X_val, y_train, y_val, n_trials=15):
    def objective(trial):
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
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_lgbm(X_train, X_val, y_train, y_val, n_trials=15):
    def objective(trial):
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
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# --- STEP 8: Valutazione modelli ---
def evaluate_models(models, X_train, X_val, y_train, y_val):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}")
        results[name] = {'MAE': mae, 'RMSE': rmse}
    return results

# --- STEP 9: Creazione submission ---
def create_submission(model, X_test, ids, output_path='csv/submission.csv'):
    preds = model.predict(X_test)
    pd.DataFrame({'id': ids, 'Price': preds}).to_csv(output_path, index=False)
    print(f"Submission salvata: {output_path}")

# --- MAIN ---
if __name__ == '__main__':
    cols_to_fill = ["Brand","Material","Size","Compartments",
                    "Laptop Compartment","Waterproof","Style",
                    "Color","Weight Capacity (kg)"]
    features = cols_to_fill.copy()

    # Load o pulizia train (only encoders saved)
    train_df, encoders, imputers = load_or_prepare_clean_data(
        TRAIN_PATH, CLEAN_TRAIN_CSV, cols_to_fill
    )
    # Load o pulizia test
    test_df = load_or_prepare_clean_test(
        TEST_PATH, CLEAN_TEST_CSV, encoders, imputers, cols_to_fill
    )

    # Preprocessing e scaling
    X_train, X_val, y_train, y_val, X_test, test_ids = prepare_and_scale_features(
        train_df, test_df, features
    )

    # Hyperparam tuning
    best_xgb = tune_xgb(X_train, X_val, y_train, y_val)
    best_lgb = tune_lgbm(X_train, X_val, y_train, y_val)

    # Costruzione e valutazione modelli
    models = {
        'XGB': XGBRegressor(**best_xgb, random_state=42),
        'LightGBM': LGBMRegressor(**best_lgb, random_state=42)
    }
    results = evaluate_models(models, X_train, X_val, y_train, y_val)
    best_name = min(results, key=lambda k: results[k]['RMSE'])
    print(f"Best model: {best_name}")

    # Creazione submission
    create_submission(models[best_name], X_test, test_ids)



