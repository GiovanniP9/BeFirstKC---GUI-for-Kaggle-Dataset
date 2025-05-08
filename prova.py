import os
import numpy as np
import pandas as pd
import optuna
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# --- 1) Pulizia e imputazione dei dati ---
def load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill):
    if os.path.exists(cleaned_path):
        print("Caricamento del dataset pulito...")
        return pd.read_csv(cleaned_path)
    print("Dataset non pulito trovato. Eseguo imputazione...")
    raw_df = pd.read_csv(train_path)
    encoded_df, _ = encode_categoricals(raw_df.copy())
    cleaned_df = impute_missing_values(raw_df, encoded_df, cols_to_fill)
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Dataset pulito salvato in: {cleaned_path}")
    return cleaned_df

def encode_categoricals(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df, None

def impute_missing_values(orig_df, enc_df, cols_to_fill):
    for col in cols_to_fill:
        if orig_df[col].isnull().sum() == 0:
            continue
        print(f"Imputazione colonna: {col}")
        y = orig_df[col]
        X = enc_df.drop(columns=[col])
        X_known, X_missing = X[y.notnull()], X[y.isnull()]
        y_known = y[y.notnull()]
        if orig_df[col].dtype == 'object':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            le = LabelEncoder()
            y_enc = le.fit_transform(y_known.astype(str))
            clf.fit(X_known, y_enc)
            y_pred = le.inverse_transform(clf.predict(X_missing))
        else:
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_known, y_known)
            y_pred = reg.predict(X_missing)
        orig_df.loc[y.isnull(), col] = y_pred
    return orig_df

# --- 2) Feature Engineering aggiuntiva ---
def create_features(df):
    # 1) Binary encoding
    df['Has_Laptop']    = df['Laptop Compartment'].map({'Yes':1, 'No':0})
    df['Is_Waterproof'] = df['Waterproof'].map({'Yes':1, 'No':0})

    # 2) Ordinal mapping per Size
    size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
    df['Size_ord'] = df['Size'].map(size_map).fillna(-1)

    # 3) Ratio features (con ε per evitare zero-division)
    eps = 1e-3
    df['Compartments_per_kg']    = df['Compartments'] / (df['Weight Capacity (kg)'] + eps)
    df['Weight_per_compartment'] = df['Weight Capacity (kg)'] / (df['Compartments'] + eps)

    # 4) Frequency encoding Brand
    df['Brand_freq']  = df.groupby('Brand')['id'].transform('count') / len(df)

    # 5) Leather flag
    df['Is_Leather'] = (df['Material'] == 'Leather').astype(int)

    # 6) Colori chiari vs scuri
    light_colors = {'White','Yellow','Beige','Pink'}
    df['Color_light'] = df['Color'].isin(light_colors).astype(int)

    return df

# --- 3) Selezione feature per VIF e p-value ---
def elimina_variabili_vif_pvalue(X_train, y_train, vif_threshold=10.0, pvalue_threshold=0.05):
    """
    Rimuove variabili da X_train basandosi su VIF e p-value:
    - elimina solo quelle con VIF > vif_threshold e p-value > pvalue_threshold
    - ricalcola iterativamente dopo ogni rimozione
    """
    # 1) Copia e casting a float
    X_current = X_train.copy().astype(float)
    y_current = y_train.astype(float)

    # 2) Aggiungi costante
    X_const = sm.add_constant(X_current).astype(float)

    while True:
        # Fit OLS su array numeric
        model  = sm.OLS(y_current.values, X_const.values).fit()
        pvals  = pd.Series(model.pvalues, index=X_const.columns).drop('const')

        # Calcola VIF
        vif_df = pd.DataFrame({
            'Feature': X_current.columns,
            'VIF': [variance_inflation_factor(X_current.values, i)
                    for i in range(X_current.shape[1])]
        })
        vif_df['p-value'] = pvals.loc[vif_df['Feature']].values

        # Filtra quelle da eliminare
        to_drop = vif_df[(vif_df['VIF'] > vif_threshold) & (vif_df['p-value'] > pvalue_threshold)]
        if to_drop.empty:
            break

        worst = to_drop.sort_values('VIF', ascending=False).iloc[0]
        feat  = worst['Feature']
        print(f"Rimuovo '{feat}' (VIF={worst['VIF']:.2f}, p-value={worst['p-value']:.4f})")

        X_current = X_current.drop(columns=[feat])
        X_const   = sm.add_constant(X_current).astype(float)

    print("Selezione completata. Feature finali:", list(X_current.columns))
    return X_current

# --- 4) Preparazione, encoding, selezione e scaling delle features ---
def prepare_and_scale_features(train_df, test_df, features):
    # Estrai raw features e target
    X_train_raw = train_df[features]
    y_train     = train_df['Price']
    X_test_raw  = test_df[features]
    test_ids    = test_df['id']

    # One-hot encoding combinato per coerenza
    X_all       = pd.concat([X_train_raw, X_test_raw], keys=['train','test'])
    X_all_dummy = pd.get_dummies(X_all)
    X_tr_all    = X_all_dummy.xs('train')
    X_te_all    = X_all_dummy.xs('test')

    # Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_all, y_train, test_size=0.2, random_state=42
    )

    # Selezione iterativa VIF + p-value su X_tr
    X_tr_sel = elimina_variabili_vif_pvalue(X_tr, y_tr,
                                             vif_threshold=10.0,
                                             pvalue_threshold=0.05)
    selected_feats = X_tr_sel.columns.tolist()

    # Applica la stessa selezione a validation e test
    X_val_sel = X_val[selected_feats]
    X_te_sel  = X_te_all[selected_feats]

    # Scaling standard
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_tr_sel)
    X_val_scaled   = scaler.transform(X_val_sel)
    X_test_scaled  = scaler.transform(X_te_sel)

    return X_train_scaled, X_val_scaled, y_tr, y_val, X_test_scaled, test_ids

# --- 5) Tuning con Optuna ---
def tune_xgb(X_train, X_val, y_train, y_val, n_trials=15):
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 50, 300),
            'max_depth':         trial.suggest_int('max_depth', 3, 12),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma':             trial.suggest_float('gamma', 0, 5),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0, 5)
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
            'n_estimators':      trial.suggest_int('n_estimators', 50, 300),
            'num_leaves':        trial.suggest_int('num_leaves', 20, 150),
            'max_depth':         trial.suggest_int('max_depth', 3, 12),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0, 5)
        }
        model = LGBMRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# --- 6) Valutazione modelli ---
def evaluate_models(models, X_train, X_val, y_train, y_val):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae   = mean_absolute_error(y_val, preds)
        rmse  = np.sqrt(mean_squared_error(y_val, preds))
        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}")
        results[name] = {'MAE': mae, 'RMSE': rmse}
    return results

# --- 7) Creazione submission ---
def create_submission(model, X_test, test_ids, output_path='submission.csv'):
    preds = model.predict(X_test)
    pd.DataFrame({'id': test_ids, 'Price': preds}).to_csv(output_path, index=False)
    print(f"Submission salvata in: {output_path}")

# --- MAIN ---
if __name__ == '__main__':
    train_path   = 'csv/train.csv'
    test_path    = 'csv/test.csv'
    cleaned_path = 'csv/train_cleaned.csv'

    # Colonne originali da imputare
    cols_to_fill = [
        "Brand","Material","Size","Compartments",
        "Laptop Compartment","Waterproof","Style",
        "Color","Weight Capacity (kg)"
    ]

    # 1) Caricamento e pulizia
    train_df = load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill)
    test_df  = pd.read_csv(test_path)

    # 2) Feature engineering
    train_df = create_features(train_df)
    test_df  = create_features(test_df)

    # 3) Preparazione, selezione e scaling
    features = cols_to_fill + [
        'Has_Laptop','Is_Waterproof','Size_ord',
        'Compartments_per_kg','Weight_per_compartment',
        'Brand_freq','Is_Leather','Color_light'
    ]
    X_train, X_val, y_train, y_val, X_test, test_ids = prepare_and_scale_features(
        train_df, test_df, features
    )

    # 4) Tuning hyperparametri
    best_xgb = tune_xgb(X_train, X_val, y_train, y_val)
    best_lgb = tune_lgbm(X_train, X_val, y_train, y_val)

    # 5) Costruzione e valutazione modelli
    models = {
        'XGB':           XGBRegressor(**best_xgb, random_state=42),
        'LightGBM':      LGBMRegressor(**best_lgb, random_state=42)
    }
    results   = evaluate_models(models, X_train, X_val, y_train, y_val)
    best_name = min(results, key=lambda k: results[k]['RMSE'])
    print(f"Best model: {best_name}")
    best_model = models[best_name]

    # 6) Creazione submission
    create_submission(best_model, X_test, test_ids)
