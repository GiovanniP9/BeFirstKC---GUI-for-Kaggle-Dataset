import os
import numpy as np
import pandas as pd
import optuna
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# --- 1) Pulizia e imputazione ---
def load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill):
    if os.path.exists(cleaned_path):
        print("Caricamento del dataset pulito...")
        return pd.read_csv(cleaned_path)
    print("Imputazione dei dati mancanti...")
    raw = pd.read_csv(train_path)
    enc, _ = encode_categoricals(raw.copy())
    clean = impute_missing_values(raw, enc, cols_to_fill)
    clean.to_csv(cleaned_path, index=False)
    return clean

def encode_categoricals(df):
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].fillna("missing")
        le   = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    return df, None

def impute_missing_values(orig, enc, cols):
    for col in cols:
        if orig[col].isnull().sum() == 0:
            continue
        y = orig[col]
        X = enc.drop(columns=[col])
        Xk, Xm = X[y.notnull()], X[y.isnull()]
        yk     = y[y.notnull()]
        if orig[col].dtype == 'object':
            from sklearn.ensemble import RandomForestClassifier
            clf   = RandomForestClassifier(n_estimators=100, random_state=42)
            le    = LabelEncoder()
            y_enc = le.fit_transform(yk.astype(str))
            clf.fit(Xk, y_enc)
            pred = le.inverse_transform(clf.predict(Xm))
        else:
            from sklearn.ensemble import RandomForestRegressor
            reg  = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(Xk, yk)
            pred = reg.predict(Xm)
        orig.loc[y.isnull(), col] = pred
    return orig

# --- 2) Feature engineering sofisticata ---
def create_features(df):
    for c in ['Material','Color']:
        vc   = df[c].value_counts(normalize=True)
        rare = vc[vc < 0.01].index
        df[c] = df[c].replace(rare, 'Other')
    df['Has_Laptop']    = df['Laptop Compartment'].map({'Yes':1,'No':0})
    df['Is_Waterproof'] = df['Waterproof'].map({'Yes':1,'No':0})
    df['Size_ord']      = df['Size'].map({'Small':0,'Medium':1,'Large':2}).fillna(-1)
    eps = 1e-3
    df['Compartments_per_kg']    = df['Compartments'] / (df['Weight Capacity (kg)'] + eps)
    df['Weight_per_compartment'] = df['Weight Capacity (kg)'] / (df['Compartments'] + eps)
    df['Is_Leather']   = (df['Material']=='Leather').astype(int)
    light = {'White','Yellow','Beige','Pink'}
    df['Color_light']  = df['Color'].isin(light).astype(int)
    return df

# --- 3) Prepara dataset + target‐encoding + log‐transform + fillna fix ---
def prepare_data(train_df, test_df, cols_to_fill):
    train_df = create_features(train_df)
    test_df  = create_features(test_df)

    train_df['LogPrice'] = np.log1p(train_df['Price'])

    base = cols_to_fill + [
        'Has_Laptop','Is_Waterproof','Size_ord',
        'Compartments_per_kg','Weight_per_compartment',
        'Is_Leather','Color_light'
    ]
    train_ohe = pd.get_dummies(train_df[base], drop_first=True)
    test_ohe  = pd.get_dummies(test_df[base], drop_first=True)
    train_ohe, test_ohe = train_ohe.align(test_ohe, join='left', axis=1, fill_value=0)

    te = TargetEncoder(cols=['Brand','Style'], smoothing=0.3)
    te.fit(train_df[['Brand','Style']], train_df['LogPrice'])
    train_te = te.transform(train_df[['Brand','Style']])
    test_te  = te.transform(test_df[['Brand','Style']])

    X_full = pd.concat([train_ohe, train_te], axis=1)
    X_test = pd.concat([test_ohe,  test_te ], axis=1)
    y_full = train_df['LogPrice']

    # Rimuove eventuali NaN residui
    X_full.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    return X_full, y_full, X_test, test_df['id']

# --- 4) Hyperparam tuning con K-Fold CV ---
def tune_model_cv(Model, X, y, param_space, n_trials=30, n_splits=4):
    def objective(trial):
        params = {}
        for k, v in param_space.items():
            if isinstance(v, tuple) and all(isinstance(x, int) for x in v):
                params[k] = trial.suggest_int(k, v[0], v[1])
            else:
                params[k] = trial.suggest_float(k, v[0], v[1])
        model = Model(**params, random_state=42)
        kf    = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X, y,
            cv=kf,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        return -scores.mean()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# --- 5) MAIN ---
if __name__ == '__main__':
    train_path   = 'csv/train.csv'
    test_path    = 'csv/test.csv'
    cleaned_path = 'csv/train_cleaned.csv'

    cols_to_fill = [
        "Brand","Material","Size","Compartments",
        "Laptop Compartment","Waterproof","Style",
        "Color","Weight Capacity (kg)"
    ]

    # Carica e pulisci
    train_df = load_or_prepare_clean_data(train_path, cleaned_path, cols_to_fill)
    test_df  = pd.read_csv(test_path)

    # Prepara features ed encoding
    X_full, y_full, X_test, test_ids = prepare_data(train_df, test_df, cols_to_fill)

    # Hold-out per valutazione finale
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    # PolynomialFeatures + Scaling
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_tr_poly   = poly.fit_transform(X_tr)
    X_val_poly  = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)

    scaler = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr_poly)
    X_val_s  = scaler.transform(X_val_poly)
    X_test_s = scaler.transform(X_test_poly)

    # Spazi di ricerca
    xgb_space = {
        'n_estimators':    (50, 500),
        'max_depth':       (3, 12),
        'learning_rate':   (0.01, 0.2),
        'subsample':       (0.6, 1.0),
        'colsample_bytree':(0.6, 1.0),
        'gamma':           (0, 3),
        'reg_alpha':       (0, 3),
        'reg_lambda':      (0, 3)
    }
    lgb_space = {
        'n_estimators':    (50, 500),
        'num_leaves':      (20, 100),
        'max_depth':       (3, 12),
        'learning_rate':   (0.01, 0.2),
        'subsample':       (0.6, 1.0),
        'colsample_bytree':(0.6, 1.0),
        'reg_alpha':       (0, 3),
        'reg_lambda':      (0, 3)
    }

    # Tuning su train set
    best_xgb = tune_model_cv(XGBRegressor, X_tr_s, y_tr, xgb_space, n_trials=4, n_splits=4)
    best_lgb = tune_model_cv(LGBMRegressor, X_tr_s, y_tr, lgb_space, n_trials=4, n_splits=4)

    # Fit e valutazione su hold-out per XGB
    xgb = XGBRegressor(**best_xgb, random_state=42)
    xgb.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
    preds_xgb = xgb.predict(X_val_s)
    rmse_xgb  = np.sqrt(mean_squared_error(y_val, preds_xgb))
    print(f"XGB RMSE on hold-out: {rmse_xgb:.3f}")

    # Fit e valutazione su hold-out per LGBM (senza early stopping)
    lgb = LGBMRegressor(**best_lgb, random_state=42)
    lgb.fit(X_tr_s, y_tr)  
    preds_lgb = lgb.predict(X_val_s)
    rmse_lgb  = np.sqrt(mean_squared_error(y_val, preds_lgb))
    print(f"LGBM RMSE on hold-out: {rmse_lgb:.3f}")

    # Scegli il migliore e crea submission
    if rmse_xgb <= rmse_lgb:
        best_model = XGBRegressor(**best_xgb, random_state=42)
        print("Selezionato XGB per la submission.")
    else:
        best_model = LGBMRegressor(**best_lgb, random_state=42)
        print("Selezionato LGBM per la submission.")

    # Retrain su tutto
    X_full_poly = poly.fit_transform(X_full)
    X_full_s    = scaler.fit_transform(X_full_poly)
    best_model.fit(X_full_s, y_full)

    X_test_poly_full = poly.transform(X_test)
    X_test_s_full    = scaler.transform(X_test_poly_full)
    preds_log        = best_model.predict(X_test_s_full)
    preds            = np.expm1(preds_log)

    pd.DataFrame({'id': test_ids, 'Price': preds}) \
      .to_csv('submission.csv', index=False)
    print("Submission salvata in: submission.csv")

