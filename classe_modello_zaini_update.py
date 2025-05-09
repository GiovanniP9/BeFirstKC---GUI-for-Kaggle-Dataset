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
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class DataCleaner:
    def __init__(self, train_path, test_path, clean_train_path, clean_test_path, encoders_path, cols_to_fill):
        self.train_path = train_path
        self.test_path = test_path
        self.clean_train_path = clean_train_path
        self.clean_test_path = clean_test_path
        self.encoders_path = encoders_path
        self.cols_to_fill = cols_to_fill
        self.encoders = None
        self.imputers = {}

    def _encode_categoricals(self, df, fit=True):
        encs = {} if fit else self.encoders
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna("missing").astype(str)
            if fit:
                le = LabelEncoder().fit(df[col])
                df[col] = le.transform(df[col])
                encs[col] = le
                print(f"[encode] '{col}' fitted: {len(le.classes_)} classes")
            else:
                df[col] = encs[col].transform(df[col])
                print(f"[encode] '{col}' transformed")
        if fit:
            self.encoders = encs
            joblib.dump(encs, self.encoders_path)
        return df

    def _train_imputers(self, df_raw, df_enc):
        exclude = ['id', 'Price']
        for col in self.cols_to_fill:
            mask_ok = df_raw[col].notnull()
            X = df_enc.loc[mask_ok].drop(columns=[c for c in exclude + [col] if c in df_enc])
            y = df_raw.loc[mask_ok, col]
            if df_raw[col].dtype == object or (col in df_enc.select_dtypes(include=['int', 'float']).columns and y.dtype == object):
                le_y = LabelEncoder().fit(y.astype(str))
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X, le_y.transform(y.astype(str)))
                self.imputers[col] = ('clf', clf, le_y)
                print(f"[imputer] '{col}' classifier with {len(le_y.classes_)} classes")
            else:
                reg = RandomForestRegressor(n_estimators=100, random_state=42)
                reg.fit(X, y)
                self.imputers[col] = ('reg', reg, None)
                print(f"[imputer] '{col}' regressor")
        joblib.dump(self.imputers, os.path.splitext(self.encoders_path)[0] + '_imputers.pkl')

    def _apply_imputers(self, df):
        exclude = ['id', 'Price']
        enc = df.copy()
        for col, le in self.encoders.items():
            enc[col] = enc[col].fillna("missing").astype(str)
            enc[col] = le.transform(enc[col])
        for col, (kind, model, le_y) in self.imputers.items():
            mask = df[col].isnull()
            if not mask.any():
                continue
            Xm = enc.loc[mask].drop(columns=[c for c in exclude + [col] if c in enc])
            if kind == 'clf':
                df.loc[mask, col] = le_y.inverse_transform(model.predict(Xm))
            else:
                df.loc[mask, col] = model.predict(Xm)
            print(f"[impute] '{col}': filled {mask.sum()} missing")
        return df

    def prepare_train(self):
        if os.path.exists(self.clean_train_path) and os.path.exists(self.encoders_path):
            print("Loading cleaned train and encoders")
            df = pd.read_csv(self.clean_train_path)
            self.encoders = joblib.load(self.encoders_path)
            imputers_file = os.path.splitext(self.encoders_path)[0] + '_imputers.pkl'
            if os.path.exists(imputers_file):
                self.imputers = joblib.load(imputers_file)
            return df

        print("Cleaning train from scratch")
        raw = pd.read_csv(self.train_path)
        enc = self._encode_categoricals(raw.copy(), fit=True)
        self._train_imputers(raw.copy(), enc)
        clean = self._apply_imputers(raw.copy())
        os.makedirs(os.path.dirname(self.clean_train_path), exist_ok=True)
        clean.to_csv(self.clean_train_path, index=False)
        return clean

    def prepare_test(self):
        if os.path.exists(self.clean_test_path):
            print("Loading cleaned test")
            return pd.read_csv(self.clean_test_path)
        print("Cleaning test from scratch")
        raw = pd.read_csv(self.test_path)
        clean = self._apply_imputers(raw.copy())
        os.makedirs(os.path.dirname(self.clean_test_path), exist_ok=True)
        clean.to_csv(self.clean_test_path, index=False)
        return clean


class FeaturePreprocessor:
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def transform(self, train_df, test_df):
        X_train = train_df[self.features]
        X_test = test_df[self.features]
        X = pd.concat([X_train, X_test], keys=['train', 'test'])
        X_d = pd.get_dummies(X)
        X_tr = X_d.xs('train')
        X_te = X_d.xs('test')
        X_tr_split, X_val_split, y_tr, y_val = train_test_split(
            X_tr, train_df['Price'], test_size=0.2, random_state=42)
        X_tr_s = self.scaler.fit_transform(X_tr_split)
        X_val_s = self.scaler.transform(X_val_split)
        X_test_s = self.scaler.transform(X_te)
        return X_tr_s, X_val_s, y_tr, y_val, X_test_s, test_df['id']


class HyperparameterTuner:
    @staticmethod
    def tune(study_name, model_cls, X_train, y_train, X_val, y_val, param_space, n_trials=15):
        def objective(trial):
            params = {k: getattr(trial, f"suggest_{v['type']}")(k, *v['bounds']) for k, v in param_space.items()}
            model = model_cls(**params, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(direction='minimize', study_name=study_name)
        study.optimize(objective, n_trials=n_trials)
        print(f"Best params for {study_name}: {study.best_params}")
        return study.best_params


class ModelEvaluator:
    @staticmethod
    def evaluate(models, X_train, y_train, X_val, y_val):
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}")
            results[name] = {'MAE': mae, 'RMSE': rmse}
        return results


class DeepLearning:
    def __init__(self, input_dim, layers_units=[128, 64, 32], dropout_rates=[0.3, 0.2, 0.1], 
                 activation='relu', learning_rate=0.001, epochs=100, batch_size=32, 
                 patience=10, model_path='models/dl_model'):
        self.input_dim = input_dim
        self.layers_units = layers_units
        self.dropout_rates = dropout_rates
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model_path = model_path
        self.model = None
        self.history = None
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.layers_units[0], input_dim=self.input_dim, 
                        activation=self.activation))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rates[0]))
        
        for i in range(1, len(self.layers_units)):
            model.add(Dense(self.layers_units[i], activation=self.activation))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rates[i]))
        
        model.add(Dense(1))
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print(f"Modello di deep learning costruito con {len(self.layers_units)} livelli nascosti")
        model.summary()
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        callbacks = []
        validation_data = None
        
        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            validation_data = (X_val, y_val)
        
        print(f"Addestramento modello deep learning con {self.epochs} epoche max...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Addestramento completato.")
        return self.history
    
    def predict(self, X):
        return self.model.predict(X).flatten()
    
    def evaluate(self, X, y):
        loss, mae = self.model.evaluate(X, y, verbose=0)
        rmse = np.sqrt(loss)
        print(f"Valutazione: MAE={mae:.3f}, RMSE={rmse:.3f}")
        return {'MAE': mae, 'RMSE': rmse}
    
    def save(self, custom_path=None):
        path = custom_path if custom_path else self.model_path
        self.model.save(path)
        print(f"Modello salvato in: {path}")
    
    def load(self, custom_path=None):
        path = custom_path if custom_path else self.model_path
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path)
            print(f"Modello caricato da: {path}")
        else:
            raise FileNotFoundError(f"Nessun modello trovato in: {path}")


class Pipeline:
    def __init__(self):
        self.cols_to_fill = ["Brand","Material","Size","Compartments",
                              "Laptop Compartment","Waterproof","Style",
                              "Color","Weight Capacity (kg)"]
        self.features = self.cols_to_fill.copy()
        self.cleaner = DataCleaner(
            train_path='csv/train.csv',
            test_path='csv/test.csv',
            clean_train_path='csv/train_cleaned.csv',
            clean_test_path='csv/test_cleaned.csv',
            encoders_path='models/encoders.pkl',
            cols_to_fill=self.cols_to_fill
        )
        self.preprocessor = FeaturePreprocessor(self.features)

    def run(self):
        # Carica training e test puliti
        train_df = self.cleaner.prepare_train()
        test_df = self.cleaner.prepare_test()

        # Carica dataset extra imputed per il modello deep learning
        extra_df = pd.read_csv('csv/train_extra_imputed.csv')

        # Preprocess per modelli non DL
        X_train, X_val, y_train, y_val, X_test, ids = \
            self.preprocessor.transform(train_df, test_df)

        # Preprocess per modello DL usando extra_df
        X_train_dl, X_val_dl, y_train_dl, y_val_dl, _, _ = \
            self.preprocessor.transform(extra_df, test_df)

        print("\n=== Addestramento Deep Learning con train_extra_imputed.csv ===")
        dl_model = DeepLearning(
            input_dim=X_train_dl.shape[1],
            layers_units=[128, 64, 32],
            dropout_rates=[0.3, 0.2, 0.1],
            epochs=50,
            batch_size=32,
            patience=10,
            model_path='models/dl_model'
        )
        dl_model.fit(X_train_dl, y_train_dl, X_val_dl, y_val_dl)
        dl_results = dl_model.evaluate(X_val_dl, y_val_dl)

        print("\n=== Ottimizzazione XGBoost ===")
        xgb_space = {
            'n_estimators': {'type': 'int', 'bounds': (50, 300)},
            'max_depth': {'type': 'int', 'bounds': (3, 12)},
            'learning_rate': {'type': 'float', 'bounds': (0.01, 0.3)},
            'subsample': {'type': 'float', 'bounds': (0.5, 1.0)},
            'colsample_bytree': {'type': 'float', 'bounds': (0.5, 1.0)},
            'gamma': {'type': 'float', 'bounds': (0, 5)},
            'reg_alpha': {'type': 'float', 'bounds': (0, 5)},
            'reg_lambda': {'type': 'float', 'bounds': (0, 5)},
        }
        best_xgb = HyperparameterTuner.tune(
            'xgb_study', XGBRegressor, X_train, y_train, X_val, y_val, xgb_space)

        print("\n=== Ottimizzazione LightGBM ===")
        lgb_space = {
            'n_estimators': {'type': 'int', 'bounds': (50, 300)},
            'num_leaves': {'type': 'int', 'bounds': (20, 150)},
            'max_depth': {'type': 'int', 'bounds': (3, 12)},
            'learning_rate': {'type': 'float', 'bounds': (0.01, 0.3)},
            'subsample': {'type': 'float', 'bounds': (0.5, 1.0)},
            'colsample_bytree': {'type': 'float', 'bounds': (0.5, 1.0)},
            'reg_alpha': {'type': 'float', 'bounds': (0, 5)},
            'reg_lambda': {'type': 'float', 'bounds': (0, 5)},
        }
        best_lgb = HyperparameterTuner.tune(
            'lgb_study', LGBMRegressor, X_train, y_train, X_val, y_val, lgb_space)

        class DLWrapper:
            def __init__(self, dl_model):
                self.dl_model = dl_model
            def fit(self, X, y): pass
            def predict(self, X): return self.dl_model.predict(X)

        print("\n=== Valutazione comparativa dei modelli ===")
        models = {
            'XGB': XGBRegressor(**best_xgb, random_state=42),
            'LightGBM': LGBMRegressor(**best_lgb, random_state=42),
            'DeepLearning': DLWrapper(dl_model)
        }
        results = ModelEvaluator.evaluate(models, X_train, y_train, X_val, y_val)
        best_name = min(results, key=lambda k: results[k]['RMSE'])
        print(f"Il miglior modello è {best_name}")

        if best_name == 'DeepLearning':
            preds = dl_model.predict(X_test)
            dl_model.save()
        else:
            preds = models[best_name].predict(X_test)

        submission = pd.DataFrame({'id': ids, 'Price': preds})
        os.makedirs('csv', exist_ok=True)
        submission.to_csv('csv/submission.csv', index=False)
        print("Submission salvata in csv/submission.csv")


if __name__ == '__main__':
    Pipeline().run()
